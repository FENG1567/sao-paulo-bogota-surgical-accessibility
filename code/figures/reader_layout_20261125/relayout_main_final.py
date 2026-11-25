"""Apply physical panel geometry to unchanged frozen-data plot artists."""
from pathlib import Path
import argparse,json,hashlib,re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pymupdf
import relayout_main as base
from figure_style import save_figure,bold_all,resolved_font

X=Path(__file__).resolve().parent
FONT=pymupdf.Font(fontfile=resolved_font)
P=argparse.ArgumentParser();P.add_argument('--figures',nargs='*',type=int,default=[1,2,3,4,5]);P.add_argument('--run',default='root01');A=P.parse_args()
OUT=X/'figures_main'/A.run
def wrap(text,width,size=8.5):
 words=' '.join(text.split()).split();lines=[];line=''
 for word in words:
  trial=(line+' '+word).strip()
  if line and FONT.text_length(trial,fontsize=size)>width*72:lines.append(line);line=word
  else:line=trial
 if line:lines.append(line)
 return '\n'.join(lines)
def pos(fig,ax,x,y,w,h):ax.set_position([x/7.2,y/fig.get_figheight(),w/7.2,h/fig.get_figheight()])
def text(fig,x,y,s,size=8.5,**kw):return fig.text(x/7.2,y/fig.get_figheight(),s,fontsize=size,fontweight='bold',ha=kw.pop('ha','left'),va=kw.pop('va','top'),**kw)
def tag(fig,letter,x,y):return text(fig,x,y,letter,18,va='bottom',gid='panel_tag_'+letter)
def clear_tags(fig):
 for t in list(fig.texts):
  if (t.get_gid() or '').startswith('panel_tag_'):t.remove()
def title(ax,s,size=10.5,width=2.8):
 ax.set_title('',loc='center');ax.set_title(wrap(s,width,size),loc='left',fontsize=size,pad=5)
def suptitle(fig):
 fig._suptitle.set_position((.045,.985));fig._suptitle.set_fontsize(12)
def legend_copy(leg):return list(leg.legend_handles),[t.get_text() for t in leg.get_texts()]
def remove_legends(fig):
 result=[legend_copy(l) for l in fig.legends]
 for l in list(fig.legends):l.remove()
 return result
def numeric_fingerprint(fig):
 values=[]
 for ax in fig.axes:
  for l in ax.lines:values.append(('line',np.asarray(l.get_xdata()).tolist(),np.asarray(l.get_ydata()).tolist()))
  for c in ax.collections:
   if hasattr(c,'get_offsets'):values.append(('offset',np.asarray(c.get_offsets()).tolist()))
   if hasattr(c,'get_segments'):values.append(('segments',[np.asarray(v).tolist() for v in c.get_segments()]))
   if hasattr(c,'get_paths'):values.append(('collection_paths',[np.asarray(p.vertices).tolist() for p in c.get_paths()]))
  for p in ax.patches:
   if hasattr(p,'get_width'):values.append(('bar_width',p.get_x(),p.get_y()+p.get_height()/2,p.get_width()))
   elif hasattr(p,'get_path'):values.append(('patch_path',np.asarray(p.get_patch_transform().transform_path(p.get_path()).vertices).tolist()))
 return hashlib.sha256(json.dumps(values,sort_keys=True,default=str).encode()).hexdigest()
def fig1(fig):
 fig.set_size_inches(7.2,4.2);a,b=fig.axes
 pos(fig,a,.72,1.18,2.2,2.27);pos(fig,b,4.65,1.18,2.40,2.27)
 a.set_ylim(1.85,-.85)
 for p in a.patches:
  center=p.get_y()+p.get_height()/2;p.set_height(.28);p.set_y(center-.14)
 for t in a.texts:
  s=t.get_text();m=re.match(r'([\d,]+) (.*)',s)
  if m:t.set_text(m[1]+'\n'+m[2]);t.set_linespacing(1.05)
 title(a,'Analysis-origin counts',10.5,2.2);title(b,'Facility inventory counts',10.5,2.40)
 for t in a.texts:
  if 'sampled origins' in t.get_text():t.set_text(t.get_text().replace('sampled origins','sampled census tracts'))
  if 'census origins' in t.get_text():t.set_text(t.get_text().replace('census origins','grid-cell origins'))
 for t in b.get_yticklabels():t.set_text(t.get_text().replace('Retained historical','Historical'))
 b.set_yticklabels([t.get_text().replace('Retained historical','Historical') for t in b.get_yticklabels()])
 tag(fig,'A',.05,3.50);tag(fig,'B',3.65,3.50)
 notes=[t for t in fig.texts if t is not fig._suptitle and not(t.get_gid() or '').startswith('panel_tag_')]
 for t in notes:
  t.set_text(wrap('São Paulo: 339 registry entries, 89 Equipment-SUS entries and 118 historical activity-flag entries; the comparison inventories overlap at 78. Equipment-SUS uses a SUS-designated CNES 0335 laparoscope/video record (2024 equipment files). São Paulo uses 1,100 sampled census tracts; Bogotá uses all 1,057 grid cells and has 16 linked public-provider entries. Full overlaps: Table 1.',6.8))
  t.set_position((.045,.035));t.set_va('bottom');t.set_linespacing(1.13)
 suptitle(fig)
def fig2(fig):
 fig.set_size_inches(7.2,6.9);axes=fig.axes
 for i,ax in enumerate(axes):
  pos(fig,ax,.72 if i%2==0 else 4.12,3.03 if i<2 else .80,2.8,1.8 if i<2 else 1.67)
  ax.set_xlim(9,61);ax.set_xticks([10,15,30,45,60])
  if i<2:
   n='1,100' if i==0 else '1,057';city='São Paulo' if i==0 else 'Bogotá';frame='probability sample' if i==0 else 'census frame'
   title(ax,f'{city} · {frame} (n={n})',10,2.8)
  else:title(ax,('São Paulo' if i==2 else 'Bogotá')+' · coverage-gap contrast',10,2.8)
  tag(fig,'ABCD'[i],.05 if i%2==0 else 3.65,4.99 if i<2 else 2.63)
 legends=remove_legends(fig)
 h,l=legends[0]
 # Mode is encoded by color; inventory is encoded by line style.
 fig.legend([h[0],h[1]],['Car','Transit + walking (WT)'],loc='upper center',bbox_to_anchor=(.5,.965),ncol=2,fontsize=8.5,labelspacing=.12,handlelength=1.5,columnspacing=1.0,borderaxespad=0)
 style_h=[h[2],h[3],h[5]]
 style_l=['Registry (solid)',
          wrap('Equipment-SUS (São Paulo) / public-provider signal (Bogotá), dashed',2.7,8.5),
          'Historical activity (São Paulo only, dotted)']
 fig.legend(style_h,style_l,loc='upper center',bbox_to_anchor=(.5,.925),ncol=1,fontsize=8.5,labelspacing=.04,handlelength=1.5,columnspacing=.8,borderaxespad=0)
 h,l=legends[1]
 contrast_h=[h[0],h[1]]
 contrast_l=['D: Registry vs Equipment-SUS / public-provider signal',
             'D: Registry vs historical activity (São Paulo)']
 fig.legend(contrast_h,contrast_l,loc='upper center',bbox_to_anchor=(.5,.835),ncol=1,fontsize=8.5,handlelength=1.5,columnspacing=.55,borderaxespad=0,labelspacing=.04)
 ci_handles=[handle for handle,label in zip(h[2:],l[2:]) if 'CI' in label]
 if ci_handles:fig.legend(ci_handles[:1],['95% design CI at 15/30/45 min (São Paulo)'],loc='upper center',bbox_to_anchor=(.5,.795),ncol=1,fontsize=8.5,handlelength=1.5,borderaxespad=0)
 for t in fig.texts:
  if t is not fig._suptitle and not(t.get_gid() or '').startswith('panel_tag_'):
   t.set_position((.045,.026));t.set_text(wrap('D = [WT (registry) − WT (comparison)] − [Car (registry) − Car (comparison)], in percentage points. Positive D means a larger inventory contrast for WT; CI = confidence interval.',6.8));t.set_va('bottom')
 suptitle(fig)
def fig3(fig):
 fig.set_size_inches(7.2,7.0);axes=fig.axes
 bottoms=[4.98,3.00,.62];heights=[1.20,1.26,1.48]
 for i,ax in enumerate(axes[:5]):
  row=i//2;pos(fig,ax,.72 if i%2==0 else 4.12,bottoms[row],2.8,heights[row]);ax.set_xlabel('')
  ax.set_xlim(9,61);ax.set_xticks([10,15,30,45,60])
  original=ax.get_title(loc='left');title(ax,original,10,2.8)
  tag(fig,'ABCDE'[i],.05 if i%2==0 else 3.65,6.31 if row==0 else (4.39 if row==1 else 2.36))
  for t in ax.texts:
   if t.get_text().startswith('Δ30'):
    t.set_text(t.get_text().replace('Δ30 WT−foot','Δ30').replace(' [','\n['));t.set_position((.025,.94));t.set_ha('left');t.set_va('top');t.set_linespacing(1.0)
 axes[4].set_yticks([0,25,50,75,100])
 for i in (1,3):
  for t in axes[i].texts:
   if t.get_text().startswith('Δ30'):t.set_position((.975,.06));t.set_ha('right');t.set_va('bottom')
 f=axes[5];pos(fig,f,4.12,.35,2.85,2.25);tag(fig,'F',3.65,2.36)
 notes='WT = transit + walking; it includes direct walking. Δ30 = 30-min WT minus direct-walking coverage (pp, percentage points). Walking-distance panels filter the same modeled-route set. São Paulo: n=1,100; whiskers show 95% design CIs at 15/30/45 min and brackets give the Δ30 CI. Bogotá: complete frame, n=1,057.'
 old=f.get_legend();h,l=legend_copy(old);old.remove()
 for t in list(f.texts):t.remove()
 f.text(0,.98,'Mode key and definitions',transform=f.transAxes,fontsize=10.5,va='top',fontweight='bold')
 f.legend(h,l,loc='upper left',bbox_to_anchor=(0,.84),ncol=1,fontsize=8.5,labelspacing=.15,handlelength=1.6,borderaxespad=0)
 f.text(0,.44,wrap(notes,2.82),transform=f.transAxes,fontsize=8.5,va='top',linespacing=1.06)
 text(fig,3.6,.06,'Threshold (min)',10,ha='center',va='bottom');fig._suptitle.set_text('Direct walking, transit and walking-distance limits');suptitle(fig)
def fig4(fig):
 fig.set_size_inches(7.2,7.6);axes=fig.axes
 legends=remove_legends(fig);h,l=legends[0]
 fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,7.24/7.6),ncol=len(l),fontsize=8.5,handlelength=1.3,columnspacing=.8,borderaxespad=0)
 for i,ax in enumerate(axes[:2]):
  pos(fig,ax,.72 if i==0 else 4.12,5.75,2.8,.9);title(ax,'São Paulo · Q1−Q5' if i==0 else 'Bogotá · low−high estrato',10.5,2.8)
  if i==0:ax.set_ylabel('Coverage difference\n(percentage points)',fontsize=9.5,labelpad=4)
  tag(fig,'AB'[i],.40 if i==0 else 3.65,6.82)
 specs=[('C',.72,3.90,'São Paulo · Registry\n30 min'),('D',4.12,3.90,'Bogotá · Registry\n30 min'),('E',.72,2.25,'São Paulo · Equipment-SUS\n30 min'),('F',4.12,2.25,'Bogotá · linked public providers\n30 min'),('G',.72,.60,'São Paulo · Historical activity\n30 min')]
 for ax,(letter,left,bottom,heading) in zip(axes[2:7],specs):
  pos(fig,ax,left,bottom,2.8,1.10)
  ax.set_xlabel('');ax.set_aspect('equal',adjustable='box');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
  if letter in 'CEG':ax.set_ylabel('Latitude (°)',fontsize=8.5,labelpad=2)
  ax.set_title('',loc='center');ax.set_title(wrap(heading,2.8,10),loc='left',fontsize=10,pad=4)
  tag(fig,letter,.05 if left<2 else 3.65,5.08 if letter in 'CD' else (3.45 if letter in 'EF' else 2.00))
 map_axes=axes[2:7]
 for city_axes in (map_axes[0:1]+map_axes[2:3]+map_axes[4:5], map_axes[1:2]+map_axes[3:4]):
  points=[]
  for ax in city_axes:
   for collection in ax.collections:
    if hasattr(collection,'get_offsets'):
     offsets=np.asarray(collection.get_offsets(),dtype=float)
     if offsets.ndim==2 and offsets.shape[1]>=2:points.append(offsets[:,:2])
  if not points:raise ValueError('Figure 4 map panel has no point coordinates.')
  all_points=np.concatenate(points,axis=0)
  padding=np.maximum(np.ptp(all_points,axis=0)*0.02,1e-4)
  lower=np.nanmin(all_points,axis=0)-padding
  upper=np.nanmax(all_points,axis=0)+padding
  center=(lower+upper)/2
  span=upper-lower
  target_aspect=2.8/1.1
  if span[0]/span[1]<target_aspect:span[0]=span[1]*target_aspect
  else:span[1]=span[0]/target_aspect
  bounds=np.vstack([center-span/2,center+span/2])
  for ax in city_axes:ax.set_xlim(bounds[0,0],bounds[1,0]);ax.set_ylim(bounds[0,1],bounds[1,1])
 point_count=sum(len(c.get_offsets()) for ax in map_axes for c in ax.collections if hasattr(c,'get_offsets'))
 assert point_count==5414,('Figure 4 point-row count changed',point_count)
 f=axes[7];pos(fig,f,4.12,.30,2.86,1.78)
 notes='Maps show analyzed origin points only (São Paulo: 1,100 sampled tracts; Bogotá: 1,057 grid-cell origins); unsampled locations are unclassified. WT = transit + walking. Unknown-group origins remain in citywide coverage totals and are excluded from subgroup contrasts. Gradients are in percentage points; São Paulo whiskers show 95% design CIs.'
 for t in list(f.texts):t.remove()
 tag(fig,'H',3.65,2.00)
 f.text(0,1,'Map classes and definitions',transform=f.transAxes,fontsize=10.5,fontweight='bold',va='top')
 hm,lm=legends[1]
 # Keep all five map classes, including classes absent from an individual map.
 lm=[('Unresolved route outcome' if v=='Technically indeterminate' else v) for v in lm]
 f.legend(hm,lm,loc='upper left',bbox_to_anchor=(0,.84),ncol=2,fontsize=8.5,labelspacing=.20,columnspacing=.5,handlelength=.8,handletextpad=.3,borderaxespad=0)
 f.text(0,.48,wrap(notes,2.84),transform=f.transAxes,va='top',fontsize=8.5,linespacing=1.04)
 text(fig,3.6,.07,'Longitude (°)',10,ha='center',va='bottom');suptitle(fig)
def fig5(fig):
 fig.set_size_inches(7.2,7.8);axes=fig.axes
 for i,ax in enumerate(axes[:5]):
  row=i//2;bottom=[6.40,5.13,3.86][row];pos(fig,ax,.80 if i%2==0 else 4.25,bottom,2.6 if i%2==0 else 2.8,.65)
  ax.set_ylim(-3,108);ax.set_yticks([0,50,100]);ax.set_ylabel('');title(ax,ax.get_title(loc='left'),10.5,2.6 if i%2==0 else 2.8)
  tag(fig,'ABCDE'[i],.05 if i%2==0 else 3.65,[7.17,5.91,4.65][row])
 text(fig,.35,5.55,'Covered at 30 min (%)',10,rotation=90,ha='center',va='center')
 f=axes[5];pos(fig,f,4.25,3.45,2.8,1.22);tag(fig,'F',3.65,4.65)
 notes=next(t.get_text() for t in f.texts if t.get_text().startswith('Validation frames'))
 h,l=legend_copy(f.get_legend());f.get_legend().remove()
 for t in list(f.texts):t.remove()
 l=[v.replace('Walking + transit (WT)','Walking + transit\n(WT)') for v in l]
 f.text(0,.98,'Mode',transform=f.transAxes,fontsize=8.5,fontweight='bold',va='top')
 mode_legend=f.legend(h[:2],l[:2],loc='upper left',bbox_to_anchor=(0,.88),ncol=2,fontsize=8.5,labelspacing=.12,columnspacing=.45,handlelength=.8,handletextpad=.25,borderaxespad=0)
 f.add_artist(mode_legend)
 f.text(0,.63,'Departure scenario',transform=f.transAxes,fontsize=8.5,fontweight='bold',va='top')
 from matplotlib.patches import Patch
 departure_handles=[Patch(facecolor=h[i].get_color(),edgecolor='none',label=l[i]) for i in range(2,len(h))]
 f.legend(departure_handles,l[2:],loc='upper left',bbox_to_anchor=(0,.56),ncol=3,fontsize=8.5,labelspacing=.12,columnspacing=.45,handlelength=.8,handletextpad=.25,borderaxespad=0)
 f.text(0,.27,wrap('São Paulo validation n=220; Bogotá validation n=200. Whiskers are pointwise 95% design CIs in both cities.',2.78),transform=f.transAxes,va='top',fontsize=8.5,linespacing=1.03)
 for letter,lab,plot,ll,lw,pl,pw,tx in [('G',axes[6],axes[7],.12,2.55,2.82,1.10,.55),('H',axes[8],axes[9],4.15,1.50,5.80,1.23,4.55)]:
  pos(fig,lab,ll,.48,lw,2.40);pos(fig,plot,pl,.48,pw,2.40)
  plot.set_xlim(-2,104);plot.set_xticks([0,50,100]);plot.set_title('',loc='left');plot.set_title('',loc='center')
  for t in lab.texts:
   t.set_text(wrap(t.get_text(),lw-.03,8.5));t.set_fontsize(8.5);t.set_linespacing(1.0)
  tag(fig,letter,.05 if letter=='G' else 4.00,3.24)
  title_text='São Paulo · facility inventory alternatives' if letter=='G' else 'Bogotá · facility inventory alternatives'
  text(fig,tx,3.29,wrap(title_text,3.4 if letter=='G' else 2.6,8.5),8.5)
  replacements={
   'Registry · all-activity arm (339)':'Registry: all-activity comparator (339)',
   'Registry · claims ≥100 (339)':'Registry: ≥100-claim comparator (339)',
   'Equipment-SUS · claims ≥100 arm (89)':'Equipment-SUS: ≥100-claim comparator (89)',
   'Equipment-SUS · raw profile (90)':'Historical equipment flag (90)',
   'All activity (156; 21 outside R)':'Annual activity (156; 21 outside registry)',
   'Municipal registry (185)':'Registry: municipal code (185)',
   'Municipal Equipment-SUS (50)':'Equipment-SUS: municipal code (50)',
   'Municipal legacy activity (56)':'Historical activity: municipal code (56)',
   'Registry-linked primary (142)':'Registry-linked sites (142)',
   'Public-provider signal · primary (16)':'Public-provider signal · main analysis (16)',
   'Public-provider signal · historical (16)':'Public-provider signal · historical comparison (16)',
   'Historical capacity (158)':'Historical full facility inventory (158)',
   'Legacy activity · primary (118)':'Historical activity · main analysis (118)',
  }
  for label in lab.texts:
   value=' '.join(label.get_text().split())
   for old,new in replacements.items():value=value.replace(old,new)
   label.set_text(wrap(value,lw-.03,8.5))
 from matplotlib.lines import Line2D
 mode_color_handles=[Line2D([0],[0],marker='o',color='#4D4D4D',linewidth=0,markersize=4,label='Car'),
                     Line2D([0],[0],marker='o',color='#0F4D92',linewidth=0,markersize=4,label='WT (transit + walking)')]
 fig.legend(handles=mode_color_handles,loc='upper center',bbox_to_anchor=(.52,3.12/7.8),ncol=2,fontsize=8.5,handlelength=.8,handletextpad=.35,columnspacing=.8,borderaxespad=0)
 for t in fig.texts:
  if t.get_text()=='30-min modeled lower coverage (%)':t.set_position((.5,.015));t.set_va('bottom')
 suptitle(fig)
"""Approved Figure 2 and Figure 3 reader-layout replacements.
Call after base data artists are created. Required helpers: pos, title, tag,
clear_tags, remove_legends, legend_copy, suptitle, and text.
"""

def fig2(fig):
    """Keep A-D and move the shared key below the enlarged 2x2 plot grid."""
    fig.set_size_inches(7.2, 6.2)
    for artist in list(fig.texts):
        if (artist.get_gid() or "").startswith("panel_tag_"):
            artist.remove()
    axes = fig.axes[:4]
    for i, ax in enumerate(axes):
        x = .72 if i % 2 == 0 else 4.12
        y = 3.82 if i < 2 else 1.78
        height = 1.55 if i < 2 else 1.45
        pos(fig, ax, x, y, 2.8, height)
        ax.set_xlim(9, 61)
        ax.set_xticks([10, 15, 30, 45, 60])
        if i < 2:
            n = "1,100" if i == 0 else "1,057"
            city = "São Paulo" if i == 0 else "Bogotá"
            frame = "probability sample" if i == 0 else "census frame"
            title(ax, f"{city} · {frame} (n={n})", 10, 2.8)
        else:
            city = "São Paulo" if i == 2 else "Bogotá"
            title(ax, f"{city} · coverage-gap contrast", 10, 2.8)
        tag(fig, "ABCD"[i], .05 if i % 2 == 0 else 3.65,
            5.50 if i < 2 else 3.36)

    legends = remove_legends(fig)
    mode_scope_handles = legends[0][0]
    contrast_handles, contrast_labels = legends[1]
    left_handles = [
        mode_scope_handles[0], mode_scope_handles[1], mode_scope_handles[2],
        mode_scope_handles[3], mode_scope_handles[5],
    ]
    left_labels = [
        "Car", "Transit + walking (WT)", "Registry",
        "Equipment-SUS / public-provider",
        "Historical activity (São Paulo)",
    ]
    right_handles = []
    right_labels = []
    for handle, label in zip(contrast_handles, contrast_labels):
        value = str(label)
        if value.startswith("D:"):
            right_handles.append(handle)
            right_labels.append(wrap("D: Registry vs narrower inventory" if "Equipment-SUS" in value else "D: Registry vs historical activity",3.0,8.5))
        elif "CI" in value:
            right_handles.append(handle)
            right_labels.append("95% design CI (São Paulo)")
        elif "Technical bound" in value:
            right_handles.append(handle)
            right_labels.append("Technical bound (dash-dot; not CI)")

    fig.legend(left_handles, left_labels, loc="upper left",
               bbox_to_anchor=(.72 / 7.2, 1.13 / 6.2), fontsize=8.5,
               handlelength=1.25, handletextpad=.30, labelspacing=.12,
               borderaxespad=0)
    fig.legend(right_handles, right_labels, loc="upper left",
               bbox_to_anchor=(3.83 / 7.2, 1.13 / 6.2), fontsize=8.5,
               handlelength=1.25, handletextpad=.30, labelspacing=.12,
               borderaxespad=0)
    for artist in list(fig.texts):
        if artist is not fig._suptitle and not (
            artist.get_gid() or ""
        ).startswith("panel_tag_"):
            artist.remove()
    suptitle(fig)


def fig3(fig):
    """Reflow five data panels by city; replace old F with an unlettered key."""
    fig.set_size_inches(7.2, 5.2)
    for artist in list(fig.texts):
        if (artist.get_gid() or "").startswith("panel_tag_"):
            artist.remove()
    axes = fig.axes
    layouts = [
        (axes[0], "A", .72, 3.05, 1.92, 1.35, "São Paulo · Registry", 8.9, 1.92, 4.58),
        (axes[2], "B", 2.94, 3.05, 1.92, 1.35, "São Paulo · Equipment-SUS", 8.9, 1.92, 4.58),
        (axes[4], "C", 5.16, 3.05, 1.92, 1.35, "São Paulo · Historical activity", 8.9, 1.92, 4.58),
        (axes[1], "D", .72, 1.13, 2.80, 1.35, "Bogotá · Registry", 9.8, 2.80, 2.65),
        (axes[3], "E", 4.12, 1.13, 2.80, 1.35, "Bogotá · Public-provider signal", 9.8, 2.80, 2.65),
    ]
    for ax, letter, x, y, width, height, heading, size, wrap_width, tag_y in layouts:
        pos(fig, ax, x, y, width, height)
        ax.set_xlabel("")
        ax.set_xlim(9, 61)
        ax.set_xticks([10, 15, 30, 45, 60])
        title(ax, heading, size, wrap_width)
        tag(fig, letter, x - .40, tag_y)
        for artist in ax.texts:
            if artist.get_text().startswith("Δ30"):
                artist.set_text(artist.get_text().replace("Δ30 WT−foot", "Δ30").replace(" [", "\n["))
                artist.set_fontsize(8.5)
                artist.set_linespacing(1.0)
                if letter in ("D", "E"):
                    artist.set_position((.975, .06))
                    artist.set_ha("right")
                    artist.set_va("bottom")
                else:
                    artist.set_position((.025, .94))
                    artist.set_ha("left")
                    artist.set_va("top")
        ax.set_ylabel("Coverage (%)" if letter in ("A", "D") else "")
    axes[4].set_yticks([0, 25, 50, 75, 100])

    note_axis = axes[5]
    old_legend = note_axis.get_legend()
    handles, labels = legend_copy(old_legend)
    old_legend.remove()
    for artist in list(note_axis.texts):
        artist.remove()
    note_axis.remove()

    text(fig, 3.60, .75, "Threshold (min)", 10,
         ha="center", va="bottom")
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(.5, .45 / 5.2), ncol=3, fontsize=8.5,
               handlelength=1.35, handletextpad=.28, columnspacing=.60,
               labelspacing=.14, borderaxespad=0)
    fig._suptitle.set_text("Direct walking, transit and walking-distance limits")
    suptitle(fig)


_original_fig1, _original_fig4, _original_fig5 = fig1, fig4, fig5

def fig1(fig):
    _original_fig1(fig)
    for artist in list(fig.texts):
        if artist is not fig._suptitle:
            artist.remove()
    fig.set_size_inches(7.2, 3.45)
    a,b=fig.axes
    pos(fig,a,.72,.52,2.2,2.18)
    pos(fig,b,4.65,.52,2.40,2.18)
    tag(fig,'A',.05,2.82);tag(fig,'B',3.65,2.82)
    suptitle(fig)

def fig4(fig):
    """Reflow maps into a filled three-row layout without changing plotted data."""
    _original_fig4(fig)
    axes = list(fig.axes)
    note = axes[7]
    handles, labels = legend_copy(note.get_legend())
    note.remove()

    # Retain the top gradient panels; replace only the old map-panel tags.
    for artist in list(fig.texts):
        gid = artist.get_gid() or ''
        if gid.startswith('panel_tag_') and gid not in {'panel_tag_A', 'panel_tag_B'}:
            artist.remove()
        elif artist.get_text() == 'Longitude (°)':
            artist.set_position((.5, .55 / 7.6))

    # Row 2: three São Paulo maps. Row 3: two Bogotá maps.
    layouts = [
        (axes[2], 'C', .78, 3.42, 1.80, 1.28, 'Registry'),
        (axes[4], 'D', 2.78, 3.42, 1.80, 1.28, 'Equipment-SUS'),
        (axes[6], 'E', 4.78, 3.42, 1.80, 1.28, 'Historical activity'),
        (axes[3], 'F', .72, 1.00, 2.80, 1.28, 'Registry'),
        (axes[5], 'G', 4.12, 1.00, 2.80, 1.28, 'Public-provider signal'),
    ]
    for ax, letter, left, bottom, width, height, heading in layouts:
        pos(fig, ax, left, bottom, width, height)
        ax.set_xlabel('')
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(MaxNLocator(3 if letter in 'CDE' else 4))
        ax.yaxis.set_major_locator(MaxNLocator(3 if letter in 'CDE' else 4))
        ax.tick_params(axis='y', labelleft=letter in {'C', 'F'})
        ax.set_ylabel('Latitude (°)' if letter in {'C', 'F'} else '', fontsize=8.5, labelpad=2)
        ax.set_title('', loc='center')
        ax.set_title(wrap(heading, width, 9.5), loc='left', fontsize=9.5, pad=4)

    # Row labels preserve the 30-minute classification context without long map titles.
    text(fig, 3.60, 5.05, 'São Paulo · 30-min origin classes', 9.5, ha='center', va='bottom')
    text(fig, 3.60, 2.64, 'Bogotá · 30-min origin classes', 9.5, ha='center', va='bottom')
    for letter, x, y in [
        ('C', .45, 4.80), ('D', 2.45, 4.80), ('E', 4.45, 4.80),
        ('F', .32, 2.37), ('G', 3.72, 2.37),
    ]:
        tag(fig, letter, x, y)

    map_groups = [
        ('São Paulo', [axes[2], axes[4], axes[6]], 1.80 / 1.28),
        ('Bogotá', [axes[3], axes[5]], 2.80 / 1.28),
    ]
    map_records = []
    for city, city_axes, target_aspect in map_groups:
        points = []
        for ax in city_axes:
            for collection in ax.collections:
                if hasattr(collection, 'get_offsets'):
                    offsets = np.asarray(collection.get_offsets(), dtype=float)
                    if offsets.ndim == 2 and offsets.shape[1] >= 2:
                        points.append(offsets[:, :2])
        if not points:
            raise ValueError(f'Figure 4 {city} maps have no point coordinates.')
        all_points = np.concatenate(points, axis=0)
        padding = np.maximum(np.ptp(all_points, axis=0) * .02, 1e-4)
        lower = np.nanmin(all_points, axis=0) - padding
        upper = np.nanmax(all_points, axis=0) + padding
        center = (lower + upper) / 2
        span = upper - lower
        if span[0] / span[1] < target_aspect:
            span[0] = span[1] * target_aspect
        else:
            span[1] = span[0] / target_aspect
        bounds = np.vstack([center - span / 2, center + span / 2])
        expected_xlim = (float(bounds[0, 0]), float(bounds[1, 0]))
        expected_ylim = (float(bounds[0, 1]), float(bounds[1, 1]))
        for ax in city_axes:
            ax.set_xlim(*expected_xlim)
            ax.set_ylim(*expected_ylim)
        assert all(np.allclose(ax.get_xlim(), expected_xlim) and np.allclose(ax.get_ylim(), expected_ylim) for ax in city_axes)
        map_records.append({
            'city': city,
            'map_point_rows': int(sum(len(c.get_offsets()) for ax in city_axes for c in ax.collections if hasattr(c, 'get_offsets'))),
            'shared_xlim': [float(bounds[0, 0]), float(bounds[1, 0])],
            'shared_ylim': [float(bounds[0, 1]), float(bounds[1, 1])],
            'equal_data_aspect': all(ax.get_aspect() == 1.0 for ax in city_axes),
            'panels': [
                {'panel': letter, 'source_axis': int(axes.index(ax)), 'point_rows': int(sum(len(c.get_offsets()) for c in ax.collections if hasattr(c, 'get_offsets')))}
                for ax, letter, *_ in layouts if ax in city_axes
            ],
        })
    point_count = sum(len(c.get_offsets()) for ax in axes[2:7] for c in ax.collections if hasattr(c, 'get_offsets'))
    assert point_count == 5414, ('Figure 4 point-row count changed', point_count)
    assert map_records[0]['map_point_rows'] == 3300 and map_records[1]['map_point_rows'] == 2114, map_records
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'Figure4_map_layout_audit.json').write_text(json.dumps({
        'source_to_new_panel_mapping': {'oldC': 'C', 'oldD': 'F', 'oldE': 'D', 'oldF': 'G', 'oldG': 'E', 'oldH': 'dropped'},
        'total_point_rows': point_count,
        'all_point_records_retained': True,
        'maps': map_records,
    }, indent=2, ensure_ascii=False))
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.51, .015),
               ncol=5, fontsize=8.5, handlelength=.65, handletextpad=.25,
               columnspacing=.75, borderaxespad=0, frameon=False)

def fig5(fig):
    """Keep A-D/F-G and expand E across both columns."""
    _original_fig5(fig)
    axes = list(fig.axes)
    note = axes[5]
    all_legends = [a for a in note.get_children() if isinstance(a, __import__('matplotlib').legend.Legend)]
    mode_leg = next(l for l in all_legends if any('Car' in t.get_text() for t in l.get_texts()))
    mode_h, mode_l = legend_copy(mode_leg)
    scenario_leg = next(l for l in all_legends if any('Baseline' in t.get_text() for t in l.get_texts()))
    scenario_h, scenario_l = legend_copy(scenario_leg)
    note.remove()
    for artist in list(fig.texts):
        gid = artist.get_gid() or ''
        if gid == 'panel_tag_F':
            artist.remove()
        elif gid == 'panel_tag_G':
            artist.set_text('F')
            artist.set_gid('panel_tag_F')
        elif gid == 'panel_tag_H':
            artist.set_text('G')
            artist.set_gid('panel_tag_G')
    for legend in list(fig.legends):
        legend_labels = [t.get_text() for t in legend.get_texts()]
        if any('WT (transit + walking)' in value for value in legend_labels):
            legend.set_bbox_to_anchor((.52, 3.28 / fig.get_figheight()), transform=fig.transFigure)
    for artist in fig.texts:
        gid = artist.get_gid() or ''
        if gid in {'panel_tag_F', 'panel_tag_G'}:
            x, y = artist.get_position()
            artist.set_position((x, y - .39 / fig.get_figheight()))
        elif artist.get_text().startswith(('São Paulo · facility inventory alternatives', 'Bogotá · facility inventory alternatives')):
            x, y = artist.get_position()
            artist.set_position((x, y - .28 / fig.get_figheight()))
    # E now spans the two-column plot width; its data and CI artists are unchanged.
    pos(fig, axes[4], .80, 3.86, 6.00, .65)
    pos(fig, axes[6], .12, .48, 2.55, 2.30)
    pos(fig, axes[7], 2.82, .48, 1.10, 2.30)
    pos(fig, axes[8], 4.15, .48, 1.50, 2.30)
    pos(fig, axes[9], 5.80, .48, 1.23, 2.30)
    for artist in fig.texts:
        if (artist.get_gid() or '') == 'panel_tag_E':
            artist.set_position((.05 / 7.2, 4.60 / 7.8))
    labels = [' '.join(str(v).split()) for v in mode_l + scenario_l]
    handles = mode_h + scenario_h
    shared_legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.51, 3.55 / 7.8),
                               ncol=5, fontsize=8.5, handlelength=.8, handletextpad=.3,
                               columnspacing=.85, borderaxespad=0, frameon=False)
    for artist in shared_legend.get_texts():
        artist.set_text(' '.join(artist.get_text().split()))


# Source-row provenance follows the presentation labels after reflow.
_record_rows = base.record_rows
def _presentation_record_rows(panel,*args,**kwargs):
    if panel in {'Figure3F','Figure4H','Figure5F'}:
        return
    mapping={
        'Figure3A':'Figure3A','Figure3B':'Figure3D','Figure3C':'Figure3B','Figure3D':'Figure3E','Figure3E':'Figure3C',
        'Figure4C':'Figure4C','Figure4D':'Figure4F','Figure4E':'Figure4D','Figure4F':'Figure4G','Figure4G':'Figure4E',
        'Figure5G':'Figure5F','Figure5H':'Figure5G',
    }
    return _record_rows(mapping.get(panel,panel),*args,**kwargs)
base.record_rows = _presentation_record_rows
base.PANEL_IDENTITY['Figure3_walk_caps']=list('ABCDE')
base.PANEL_IDENTITY['Figure4_income_geography']=list('ABCDEFG')
base.PANEL_IDENTITY['Figure5_validation_sensitivity']=list('ABCDEFG')

"""Compact geometry overrides for the existing main figures 1-3.
Call in the plotting namespace before AUDIT. Existing artists/data are retained.
"""
import re
from matplotlib.text import Text as _CompactText

_compact_previous_fig1, _compact_previous_fig2, _compact_previous_fig3 = fig1, fig2, fig3


def _compact_font(fig):
    for artist in fig.findobj(match=_CompactText):
        artist.set_fontfamily("Arial")
        artist.set_fontweight("bold")


def _compact_place_tags(fig, positions):
    for artist in fig.texts:
        gid = artist.get_gid() or ""
        if gid.startswith("panel_tag_"):
            letter = gid.removeprefix("panel_tag_")
            if letter in positions:
                x, y = positions[letter]
                artist.set_position((x / 7.2, y / fig.get_figheight()))
                artist.set_fontsize(18)
                artist.set_fontfamily("Arial")
                artist.set_fontweight("bold")


def fig1(fig):
    _compact_previous_fig1(fig)
    fig.set_size_inches(7.2, 3.15)
    axes = fig.axes[:2]
    a, b = axes
    pos(fig, a, 1.05, .52, 1.75, 2.00)
    pos(fig, b, 4.65, .52, 2.35, 2.00)

    # Move the sampling unit beside each city; keep the bar-end annotation numeric.
    unit_by_city = {
        "S\u00e3o Paulo": "S\u00e3o Paulo\nsampled\ncensus tracts",
        "Bogot\u00e1": "Bogot\u00e1\ngrid-cell\norigins",
    }
    old_labels = [label.get_text() for label in a.get_yticklabels()]
    a.set_yticklabels([unit_by_city.get(label, label) for label in old_labels])
    for artist in a.texts:
        match = re.match(r"^\s*([\d,]+)(?:\s|\n)", artist.get_text())
        if match:
            artist.set_text(match.group(1))

    title(a, "Analysis-origin counts", 10.5, 2.45)
    title(b, "Facility inventory counts", 10.5, 2.62)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=9.5)
        ax.xaxis.label.set_size(9.5)
        ax.yaxis.label.set_size(9.5)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontsize(9.5)
    for artist in a.texts:
        artist.set_fontsize(9.5)

    _compact_place_tags(fig, {"A": (.05, 2.73), "B": (3.65, 2.73)})
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11)
    _compact_font(fig)


def fig2(fig):
    _compact_previous_fig2(fig)
    fig.set_size_inches(7.2, 5.45)
    axes = fig.axes[:4]
    for i, ax in enumerate(axes):
        x = .72 if i % 2 == 0 else 4.12
        y = 3.38 if i < 2 else 1.55
        height = 1.48 if i < 2 else 1.42
        pos(fig, ax, x, y, 2.8, height)
        ax.set_xlim(9, 61)
        ax.set_xticks([10, 15, 30, 45, 60])
        title(ax, ax.get_title(loc="left"), 10.5, 2.8)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.xaxis.label.set_size(9.5)
        ax.yaxis.label.set_size(9.5)

    _compact_place_tags(fig, {
        "A": (.05, 5.02), "B": (3.65, 5.02),
        "C": (.05, 3.12), "D": (3.65, 3.12),
    })
    for legend, x in zip(fig.legends, (.72 / 7.2, 3.83 / 7.2)):
        legend.set_bbox_to_anchor((x, 1.05 / 5.45), transform=fig.transFigure)
        for artist in legend.get_texts():
            artist.set_fontsize(9.5)
            artist.set_fontfamily("Arial")
            artist.set_fontweight("bold")
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11)
    _compact_font(fig)


def fig3(fig):
    _compact_previous_fig3(fig)
    fig.set_size_inches(7.2, 4.85)
    axes = list(fig.axes)
    # Preserve the established mapping: ABC are S\u00e3o Paulo, DE are Bogot\u00e1.
    layouts = [
        (axes[0], "A", .72, 2.72, 2.05, 1.30, 4.31, 2.05),
        (axes[2], "B", 2.87, 2.72, 2.05, 1.30, 4.31, 2.05),
        (axes[4], "C", 5.02, 2.72, 2.05, 1.30, 4.31, 2.05),
        (axes[1], "D", .72, .91, 2.92, 1.30, 2.38, 2.92),
        (axes[3], "E", 3.98, .91, 2.92, 1.30, 2.38, 2.92),
    ]
    for ax, letter, x, y, width, height, tag_y, wrap_width in layouts:
        pos(fig, ax, x, y, width, height)
        title(ax, ax.get_title(loc="left"), 10.5, wrap_width)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.set_ylabel("Coverage (%)" if letter in ("A", "D") else "", fontsize=9.5)
        if letter in ("B", "C", "E"):
            ax.tick_params(axis="y", labelleft=False)
        for artist in ax.texts:
            if artist.get_text().startswith("\u039430"):
                artist.set_fontsize(9.5)
                artist.set_fontfamily("Arial")
                artist.set_fontweight("bold")
        ax.xaxis.label.set_size(9.5)
        ax.yaxis.label.set_size(9.5)

    _compact_place_tags(fig, {
        "A": (.32, 4.31), "B": (2.47, 4.31), "C": (4.62, 4.31),
        "D": (.32, 2.42), "E": (3.86, 2.42),
    })
    for artist in fig.texts:
        if artist.get_text() == "Threshold (min)":
            artist.set_position((.5, .52 / 4.85))
            artist.set_fontsize(9.5)
    for legend in fig.legends:
        legend.set_bbox_to_anchor((.5, .43 / 4.85), transform=fig.transFigure)
        for artist in legend.get_texts():
            artist.set_fontsize(9.5)
            artist.set_fontfamily("Arial")
            artist.set_fontweight("bold")
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11)
    _compact_font(fig)


"""Compact, data-preserving geometry overrides for main Figures 4 and 5.

Insert before ``AUDIT = []`` in the private figure driver.  The existing
layout functions remain the source for artist identity and legend handles;
this fragment only adjusts their physical placement and text styling.
"""

from matplotlib.text import Text

_compact_previous_fig4, _compact_previous_fig5 = fig4, fig5


def _compact_style_main45(fig):
    """Apply the requested Arial-bold typography after new artists are added."""
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily("Arial")
        artist.set_fontweight("bold")
        if artist.get_visible() and artist.get_fontsize() < 9.5:
            artist.set_fontsize(9.5)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11)
        fig._suptitle.set_fontfamily("Arial")
        fig._suptitle.set_fontweight("bold")
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=9.5)
        if ax.title.get_text():
            ax.title.set_fontsize(10.5)
            ax.title.set_fontweight("bold")
            ax.title.set_fontfamily("Arial")


def fig4(fig):
    """Compact Figure 4 while retaining its seven panels and every map point."""
    before = numeric_fingerprint(fig)

    # Reuse the approved seven-panel map reflow, then compact the vertical gaps.
    _compact_previous_fig4(fig)
    fig.set_size_inches(7.2, 6.50)
    axes = list(fig.axes)

    # Top evidence row: socioeconomic-gradient estimates and CIs.
    for i, ax in enumerate(axes[:2]):
        pos(fig, ax, .72 if i == 0 else 4.12, 4.80, 2.80, .75)
        title(ax, "São Paulo · Q1−Q5" if i == 0 else "Bogotá · low−high estrato",
              10.5, 2.8)
        ax.set_ylabel("Coverage difference\n(percentage points)" if i == 0 else "",
                      fontsize=9.5, labelpad=4)
        ax.tick_params(axis="both", labelsize=9.5)

    # Keep both semantic legends: gradient encoding at the top, map classes at
    # the bottom.  Their source handles and labels come from the previous layout.
    for legend in fig.legends:
        for entry in legend.get_texts():
            if entry.get_text() == "95% design CI at 30 min (SP only)":
                entry.set_text("95% design CI at 30 min (São Paulo only)")
        labels = [entry.get_text() for entry in legend.get_texts()]
        for entry in legend.get_texts():
            entry.set_fontsize(9.5)
        if any("Car" == value for value in labels):
            legend.set_bbox_to_anchor((.5, 6.18 / 6.50), transform=fig.transFigure)
        elif any("Both covered" in value for value in labels):
            legend.set_bbox_to_anchor((.51, .12 / 6.50), transform=fig.transFigure)
            legend._ncols = 5

    # Three São Paulo maps and two Bogotá maps.  Axes for a city share both
    # bounds and data aspect; bounds are recalculated for these exact rectangles.
    layouts = [
        (axes[2], "C", .80, 2.72, 1.80, 1.20, "Registry"),
        (axes[4], "D", 2.85, 2.72, 1.80, 1.20, "Equipment-SUS"),
        (axes[6], "E", 4.90, 2.72, 1.80, 1.20, "Historical activity"),
        (axes[3], "F", .72, .75, 2.80, 1.15, "Registry"),
        (axes[5], "G", 4.12, .75, 2.80, 1.15, "Public-provider signal"),
    ]
    for ax, _letter, left, bottom, width, height, heading in layouts:
        pos(fig, ax, left, bottom, width, height)
        ax.set_xlabel("")
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(3 if _letter in {"C", "D", "E"} else 4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=9.5)
        ax.tick_params(axis="y", labelleft=_letter in {"C", "F"})
        ax.set_title("", loc="center")
        ax.set_title(wrap(heading, width, 10.5), loc="left", fontsize=10.5, pad=4,
                     fontweight="bold", fontfamily="Arial")
        ax.set_ylabel("Latitude (°)" if _letter in {"C", "F"} else "",
                      fontsize=9.5, labelpad=2)

    map_axes = axes[2:7]
    map_groups = [
        ("São Paulo", map_axes[0:1] + map_axes[2:3] + map_axes[4:5], 1.80 / 1.20),
        ("Bogotá", map_axes[1:2] + map_axes[3:4], 2.80 / 1.15),
    ]
    point_count = 0
    map_records = []
    for city, city_axes, target_aspect in map_groups:
        point_arrays = []
        for ax in city_axes:
            for collection in ax.collections:
                if hasattr(collection, "get_offsets"):
                    offsets = np.ma.asarray(collection.get_offsets(), dtype=float).filled(np.nan)
                    if offsets.ndim == 2 and offsets.shape[1] >= 2:
                        point_arrays.append(offsets[:, :2])
        if not point_arrays:
            raise ValueError(f"Figure 4 {city} maps have no point coordinates.")
        points = np.concatenate(point_arrays, axis=0)
        point_count += len(points)
        padding = np.maximum(np.ptp(points, axis=0) * .02, 1e-4)
        lower = np.nanmin(points, axis=0) - padding
        upper = np.nanmax(points, axis=0) + padding
        center = (lower + upper) / 2
        span = upper - lower
        if span[0] / span[1] < target_aspect:
            span[0] = span[1] * target_aspect
        else:
            span[1] = span[0] / target_aspect
        bounds = np.vstack([center - span / 2, center + span / 2])
        for ax in city_axes:
            ax.set_xlim(float(bounds[0, 0]), float(bounds[1, 0]))
            ax.set_ylim(float(bounds[0, 1]), float(bounds[1, 1]))
            offsets = [
                np.ma.asarray(collection.get_offsets(), dtype=float).filled(np.nan)[:, :2]
                for collection in ax.collections if hasattr(collection, "get_offsets")
            ]
            offsets = [values for values in offsets if values.size]
            if offsets:
                observed = np.concatenate(offsets, axis=0)
                finite = np.isfinite(observed).all(axis=1)
                assert np.all(observed[finite, 0] >= bounds[0, 0])
                assert np.all(observed[finite, 0] <= bounds[1, 0])
                assert np.all(observed[finite, 1] >= bounds[0, 1])
                assert np.all(observed[finite, 1] <= bounds[1, 1])
        assert all(ax.get_aspect() == 1.0 for ax in city_axes)
        assert all(np.allclose(ax.get_xlim(), city_axes[0].get_xlim()) and
                   np.allclose(ax.get_ylim(), city_axes[0].get_ylim()) for ax in city_axes)
        map_records.append({
            "city": city,
            "point_rows": int(len(points)),
            "shared_xlim": [float(value) for value in city_axes[0].get_xlim()],
            "shared_ylim": [float(value) for value in city_axes[0].get_ylim()],
            "panels": [letter for ax, letter, *_ in layouts if ax in city_axes],
        })
    assert point_count == 5414, ("Figure 4 point-row count changed", point_count)
    assert numeric_fingerprint(fig) == before, "Figure 4 numeric artists changed during reflow"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "Figure4_map_layout_compact_audit.json").write_text(json.dumps({
        "total_point_rows": point_count,
        "all_point_records_retained": True,
        "maps": map_records,
    }, indent=2, ensure_ascii=False))

    # Reuse the existing row labels and replace the prior panel letters.
    for artist in fig.texts:
        if artist.get_text() == "São Paulo · 30-min origin classes":
            artist.set_position((3.60 / 7.2, 4.23 / 6.50))
            artist.set_fontsize(9.5)
        elif artist.get_text() == "Bogotá · 30-min origin classes":
            artist.set_position((3.60 / 7.2, 2.25 / 6.50))
            artist.set_fontsize(9.5)
    clear_tags(fig)
    for letter, x, y in [
        ("A", .40, 5.65), ("B", 3.70, 5.65),
        ("C", .40, 3.97), ("D", 2.50, 3.97), ("E", 4.60, 3.97),
        ("F", .40, 1.97), ("G", 3.80, 1.97),
    ]:
        tag(fig, letter, x, y)
    for artist in fig.texts:
        if artist.get_text() == "Longitude (°)":
            artist.set_position((.5, .34 / 6.50))
            artist.set_fontsize(9.5)
            artist.set_va("bottom")
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(11)
    _compact_style_main45(fig)


def fig5(fig):
    original_axes=list(fig.axes)
    before=numeric_fingerprint(fig)
    _compact_previous_fig5(fig)
    fig.set_size_inches(7.2,7.2)
    axes=original_axes
    # Validation uses marker shape for mode and color for departure scenario.
    validation=next(l for l in fig.legends if any('Baseline' in t.get_text() for t in l.get_texts()))
    sensitivity=next(l for l in fig.legends if any('WT (transit + walking)' in t.get_text() for t in l.get_texts()) and not any('Baseline' in t.get_text() for t in l.get_texts()))
    validation.set_title('')
    validation.set_bbox_to_anchor((.51,3.62/7.2),transform=fig.transFigure)
    sensitivity.set_title('')
    sensitivity.set_bbox_to_anchor((1.90/7.2,3.35/7.2),transform=fig.transFigure)
    for legend in [validation,sensitivity]:
        for t in legend.get_texts():t.set_fontsize(9.5)
    for i,ax in enumerate(axes[:4]):
        pos(fig,ax,.72 if i%2==0 else 4.12,6.02 if i<2 else 5.00,2.80,.58)
        ax.set_ylim(-3,108);ax.set_yticks([0,50,100]);ax.set_ylabel('')
        title(ax,ax.get_title(loc='left'),10.5,2.80)
        ax.tick_params(axis='both',labelsize=9.5)
    pos(fig,axes[4],.72,3.95,6.00,.58)
    axes[4].set_ylim(-3,108);axes[4].set_yticks([0,50,100]);axes[4].set_ylabel('')
    title(axes[4],axes[4].get_title(loc="left").replace("Legacy activity","Historical activity"),10.5,6.00)
    axes[4].tick_params(axis='both',labelsize=9.5)
    # Wider label strips retain every facility definition and count on one line.
    positions=[(axes[6],axes[7],.10,2.95,3.10,.90),(axes[8],axes[9],4.18,1.67,5.90,1.12)]
    for lab,plot,label_left,label_width,plot_left,plot_width in positions:
        pos(fig,lab,label_left,.45,label_width,2.40)
        pos(fig,plot,plot_left,.45,plot_width,2.40)
        plot.set_xlim(-2,104);plot.set_xticks([0,50,100]);plot.tick_params(axis='x',labelsize=9.5,pad=3)
        for label in lab.texts:
            value=' '.join(label.get_text().split())
            label.set_text(wrap(value,label_width-.03,9.5));label.set_fontsize(9.5);label.set_linespacing(1.0)
    for t in fig.texts:
        value=t.get_text()
        if value.startswith('São Paulo · facility inventory alternatives'):
            t.set_text('São Paulo · facility inventory alternatives');t.set_position((.55/7.2,3.03/7.2));t.set_fontsize(9.5)
        elif value.startswith('Bogotá · facility inventory alternatives'):
            t.set_text(wrap('Bogotá · facility inventory alternatives',2.55,9.5));t.set_position((4.55/7.2,3.03/7.2));t.set_fontsize(9.5)
        elif value=='Covered at 30 min (%)':
            t.set_position((.35/7.2,5.30/7.2));t.set_fontsize(9.5)
        elif value=='30-min modeled lower coverage (%)':
            t.set_position((.5,.07/7.2));t.set_va('bottom');t.set_fontsize(9.5)
    clear_tags(fig)
    for letter,x,y in [('A',.05,6.67),('B',3.65,6.67),('C',.05,5.65),('D',3.65,5.65),('E',.05,4.60),('F',.05,2.99),('G',4.05,2.99)]:
        tag(fig,letter,x,y)
    assert numeric_fingerprint(fig)==before,'Figure 5 numeric artists changed'
    _compact_style_main45(fig)


# Preserve numerical text annotations in addition to plotted coordinates/intervals.
def _compact_numeric_text(fig):
    values=[]
    for ax in fig.axes:
        for artist in ax.texts:
            values.extend(re.findall(r"[+−-]?\d+(?:[.,]\d+)*", artist.get_text()))
    return sorted(values)
def _compact_capture(previous):
    def apply(fig):
        previous(fig)
        fig._compact_annotation_numbers_before = _compact_numeric_text(fig)
    return apply
for _name in ['_compact_previous_fig1','_compact_previous_fig2','_compact_previous_fig3','_compact_previous_fig4','_compact_previous_fig5']:
    if _name in globals():
        globals()[_name]=_compact_capture(globals()[_name])


def reader_polish(fig, number):
 from matplotlib.container import ErrorbarContainer
 from matplotlib.ticker import FuncFormatter
 if number == 3:
  # Keep the 10-minute ticks and data; avoid repeating labels at column seams.
  for ax in (fig.axes[2], fig.axes[4]):
   ax.xaxis.set_major_formatter(FuncFormatter(lambda value, index: '' if abs(value-10)<1e-8 else f'{value:g}'))
 if number == 5:
  for i, ax in enumerate(fig.axes):
   for container in ax.containers:
    if not isinstance(container, ErrorbarContainer): continue
    point, caps, bars = container.lines
    if point is not None:
     point.set_markersize(2.6 if i < 5 else 2.2)
     point.set_zorder(3)
    for cap in caps:
     cap.set_markersize(6.0 if i < 5 else 3.4)
     cap.set_markeredgewidth(1.0)
     cap.set_zorder(4)
    for bar in bars:
     bar.set_linewidth(1.1 if i < 5 else 1.0)
     bar.set_zorder(4)
  validation = next(l for l in fig.legends if any('Baseline' in t.get_text() for t in l.get_texts()))
  handles, labels = legend_copy(validation)
  anchor=validation.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
  validation.remove()
  fig.legend(handles[:2], ['Car','Transit + walking (WT)'],loc='upper center',bbox_to_anchor=(anchor.x0,anchor.y0),ncol=2,fontsize=9.5,handlelength=1.0,handletextpad=.4,columnspacing=1.1,borderaxespad=0)
  for legend in fig.legends:
   for text in legend.get_texts():
    if text.get_text() == 'WT (transit + walking)': text.set_text('Transit + walking (WT)')

AUDIT=[]
def final_save(fig,out,stem):
 # Force original aspect/limits first; all data arrays must survive reflow.
 bold_all(fig);fig.canvas.draw();before=numeric_fingerprint(fig);clear_tags(fig)
 number=int(stem[6]);globals()['fig'+str(number)](fig);reader_polish(fig,number)
 from matplotlib.text import Text
 for artist in fig.findobj(match=Text):
  artist.set_fontfamily('Arial');artist.set_fontweight('bold')
  if (artist.get_gid() or '').startswith('panel_tag_'):artist.set_fontsize(18)
  elif artist.get_fontsize()<9.5:artist.set_fontsize(9.5)
 assert fig._compact_annotation_numbers_before==_compact_numeric_text(fig),(stem,'numerical text annotations changed')
 after=numeric_fingerprint(fig);assert before==after,(stem,'data artists changed')
 files=save_figure(fig,OUT,stem)
 AUDIT.append({'figure':stem,'before_data_artist_sha256':before,'after_data_artist_sha256':after,'data_artists_unchanged':True,'exports':files})
 print(json.dumps({'figure':stem,'status':'EXPORTED','data_artists_unchanged':True}),flush=True)
base.save_figure=final_save;base.load_sources()
for i in A.figures:getattr(base,'figure'+str(i))()
base.write_contract(set(A.figures))
OUT.mkdir(parents=True,exist_ok=True);(OUT/'layout_data_audit.json').write_text(json.dumps(AUDIT,indent=2))
