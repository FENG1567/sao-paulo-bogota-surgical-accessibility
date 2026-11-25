"""Shared typography and editable exports for a layout-only figure revision."""
from pathlib import Path
import json,hashlib,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

X=Path(__file__).resolve().parent
font_dir=Path(os.environ.get('ARIAL_FONT_DIR',str(X/'fonts')))
for name in ['arial.ttf','arialbd.ttf']:
 if (font_dir/name).is_file():font_manager.fontManager.addfont(str(font_dir/name))
resolved_font=font_manager.findfont(font_manager.FontProperties(family='Arial',weight='bold'),fallback_to_default=False)
plt.rcParams.update({
 'font.family':'Arial','font.weight':'bold','font.size':9,
 'axes.labelweight':'bold','axes.titleweight':'bold',
 'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':8.5,
 'ytick.labelsize':8.5,'legend.fontsize':8.5,'figure.titlesize':12,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.8,
 'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','text.color':'#17202A','axes.labelcolor':'#17202A',
 'xtick.color':'#17202A','ytick.color':'#17202A','figure.facecolor':'white',
 'mathtext.default':'bf','mathtext.fontset':'dejavusans',
})

def bold_all(fig):
 for obj in fig.findobj(match=Text):
  obj.set_fontfamily('Arial');obj.set_fontweight('bold')

def panel_tag(fig,ax,letter,x=None,y=None):
 pos=ax.get_position()
 return fig.text(pos.x0-0.065 if x is None else x,pos.y1+0.012 if y is None else y,
                 letter.upper(),fontsize=18,fontfamily='Arial',fontweight='bold',
                 ha='left',va='bottom',clip_on=False,gid='panel_tag_'+letter.upper())

def save_figure(fig,out,stem):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);bold_all(fig);fig.canvas.draw()
 files=[str(out/(stem+ext)) for ext in ['.pdf','.svg','.png','.tiff']]
 fig.savefig(out/(stem+'.pdf'),facecolor='white',bbox_inches=None)
 fig.savefig(out/(stem+'.svg'),facecolor='white',bbox_inches=None)
 fig.savefig(out/(stem+'.png'),facecolor='white',bbox_inches=None,dpi=300)
 fig.savefig(out/(stem+'.tiff'),facecolor='white',bbox_inches=None,dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 # Record only text actually drawn, after restoring the canvas renderer DPI.
 drawn=[];original=Text.draw
 def trace(t,renderer):
  result=original(t,renderer)
  if t.get_visible() and t.get_text().strip():
   drawn.append({'text':t.get_text(),'size':t.get_fontsize(),'weight':t.get_fontweight(),'gid':t.get_gid(),'visible':True,'bbox_pixels':list(t.get_window_extent(renderer).extents)})
  return result
 Text.draw=trace
 try:fig.canvas.draw()
 finally:Text.draw=original
 renderer=fig.canvas.get_renderer()
 record={'figure':stem,'size_inches':list(fig.get_size_inches()),'canvas_pixels':list(fig.bbox.bounds),
         'text':[{'text':t.get_text(),'size':t.get_fontsize(),'weight':t.get_fontweight(),
                  'family':t.get_fontfamily(),'gid':t.get_gid(),'visible':t.get_visible(),
                  'bbox_pixels':list(t.get_window_extent(renderer).extents)} for t in fig.findobj(match=Text) if t.get_text()],
         'drawn_text_at_consistent_dpi':drawn,'drawn_canvas_pixels':list(fig.bbox.bounds),
         'outputs':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
 (out/(stem+'.style.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2))
 plt.close(fig);return files
