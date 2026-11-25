#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# sih_extract.R -- Decode SIH-SUS reduced-AIH (.dbc) files and keep only
#                  admissions that can plausibly enter the BDI study.
#
# WHY THIS EXISTS
#   Raw SIH-SUS RD files are 3,240 .dbc files (~8.4 GB) covering 2015-2024.
#   The measured uplink to the analysis server is ~0.09 MB/s, so shipping the
#   raw corpus is impossible inside the project window. We decode locally,
#   keep only candidate admissions, and ship the extract. The extract retains
#   raw PROC_REA and all diagnosis fields, so the analytical definitions are
#   applied on the server, never baked in here.
#
# STRUCTURAL NOTE (critical for interpretation)
#   The reduced AIH ("arquivo RD") carries exactly ONE procedure per admission
#   (PROC_REA). It therefore CANNOT show a bile-duct repair performed during
#   the same admission as the cholecystectomy -- a fundamental asymmetry with
#   US NRD, which carries up to 25 procedure codes. Brazilian numerators must
#   therefore come from (a) diagnosis codes on the cholecystectomy admission,
#   or (b) a *separate* admission whose principal procedure is the repair.
#
# USAGE
#   Rscript sih_extract.R --year 2015 --rd-root <dir> --codes <codes.csv> \
#           --out <file.parquet> [--workers 10]
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(read.dbc)
  library(data.table)
  library(arrow)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  args[i + 1]
}
year    <- as.integer(getarg("--year"))
rd_root <- getarg("--rd-root")
codes_f <- getarg("--codes")
out_f   <- getarg("--out")
workers <- as.integer(getarg("--workers", "10"))

stopifnot(!is.na(year), !is.null(rd_root), !is.null(codes_f), !is.null(out_f))

# ---- code configuration ---------------------------------------------------
# The codebook is a CSV so that every definition is auditable and re-runnable.
# Columns: kind (PROC|DX), prefix, set, label
cfg <- fread(codes_f)
proc_prefix <- cfg[kind == "PROC", unique(prefix)]
dx_prefix   <- cfg[kind == "DX",   unique(prefix)]
proc_re <- paste0("^(", paste(proc_prefix, collapse = "|"), ")")
dx_re   <- paste0("^(", paste(dx_prefix,   collapse = "|"), ")")

cat(sprintf("[%d] proc prefixes=%d dx prefixes=%d\n", year,
            length(proc_prefix), length(dx_prefix)))

KEEP <- c("UF_ZI", "ANO_CMPT", "MES_CMPT", "N_AIH", "IDENT", "CNES",
          "MUNIC_RES", "MUNIC_MOV", "PROC_REA", "DIAG_PRINC", "DIAG_SECUN",
          paste0("DIAGSEC", 1:9), "CID_ASSO", "CID_MORTE",
          "IDADE", "COD_IDADE", "SEXO", "MORTE", "DIAS_PERM",
          "UTI_MES_TO", "VAL_TOT", "VAL_SH", "VAL_SP",
          "NATUREZA", "FINANC", "COMPLEX", "RACA_COR",
          "DT_INTER", "DT_SAIDA", "CAR_INT", "NUM_PROC", "ESPEC")

one_file <- function(path) {
  d <- tryCatch(read.dbc(path), error = function(e) NULL)
  if (is.null(d)) return(NULL)
  setDT(d)
  if (nrow(d) == 0) return(NULL)

  # SIGTAP codes are 10 digits (GGSSFFPPPP) but arrive as numbers, so a leading
  # zero is silently lost by the DBF reader. Left-pad before any matching, or the
  # 04.07.03.xxx biliary block is never found.
  pad10 <- function(x) {
    x <- as.character(x)
    n <- nchar(x)
    ifelse(n < 10, paste0(strrep("0", 10 - n), x), x)
  }
  d[, PROC_REA := pad10(PROC_REA)]

  # ---- candidate filter ---------------------------------------------------
  # Anchored on the PROCEDURE only. The denominator is a cholecystectomy, so an
  # admission without one cannot be in the denominator; and because SIH carries a
  # single procedure, the only reachable numerator is either a diagnosis on that
  # same admission or a repair that IS its principal procedure.
  #
  # The whole 04.07.03 biliary family was enumerated from the data (19 distinct
  # codes in 2015) and only two are reconstructive (ANASTOMOSE BILEO-DIGESTIVA,
  # COLEDOCOPLASTIA); the rest are splenectomy, cholecystostomy, choledochotomy or
  # hepatic-abscess drainage. A broad "any digestive surgery with a biliary
  # diagnosis" allowance therefore only inflates the extract without adding
  # numerator events.
  hit_pr <- grepl(proc_re, as.character(d$PROC_REA), perl = TRUE)
  keep <- hit_pr
  if (!any(keep)) return(NULL)

  out <- d[keep]
  cols <- intersect(KEEP, names(out))
  out <- out[, ..cols]
  out[, src_file := basename(path)]
  out
}

files <- list.files(file.path(rd_root, as.character(year)),
                    pattern = "\\.dbc$", recursive = TRUE, full.names = TRUE)
cat(sprintf("[%d] %d dbc files found\n", year, length(files)))
if (length(files) == 0) quit(status = 0)

t0 <- Sys.time()
cl <- makeCluster(workers)
on.exit(stopCluster(cl), add = TRUE)
clusterExport(cl, c("one_file", "dx_re", "proc_re", "KEEP"), envir = environment())
clusterEvalQ(cl, suppressPackageStartupMessages({library(read.dbc); library(data.table)}))

res <- parLapply(cl, files, function(f) {
  tryCatch(one_file(f), error = function(e) { message("FAIL ", f, ": ", conditionMessage(e)); NULL })
})
ok <- Filter(Negate(is.null), res)
cat(sprintf("[%d] decoded %d/%d files with kept rows in %.0fs\n", year,
            length(ok), length(files), as.numeric(difftime(Sys.time(), t0, units = "secs"))))

if (length(ok) == 0) quit(status = 0)
dt <- rbindlist(ok, fill = TRUE, use.names = TRUE)
cat(sprintf("[%d] candidate admissions: %d\n", year, nrow(dt)))

dir.create(dirname(out_f), recursive = TRUE, showWarnings = FALSE)
write_parquet(as_arrow_table(dt), out_f)
cat(sprintf("[%d] wrote %s (%.1f MB)\n", year, out_f, file.size(out_f) / 1e6))
