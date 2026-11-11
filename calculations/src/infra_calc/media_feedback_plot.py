"""Figure 12-4: measured-in-simulation delivery and quality, from public traces."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from .paths import PROJECT
from .sources import sha256_or_lfs_oid

DIRECTORY = PROJECT / 'figures/media-feedback'
NAMES = ('mixed-fifo-immediate', 'mixed-fifo-aggregate', 'mixed-priority-immediate', 'mixed-priority-aggregate')
ARTIFACTS = ('figure.png', 'figure.svg', 'data.json')


def digest(path):
    return sha256_or_lfs_oid(path)


def input_paths():
    return sorted({Path(__file__), PROJECT/'scenarios/book.json', PROJECT/'configs/sources.lock.json',
                   *(PROJECT/f'results/media-feedback-{name}.json' for name in NAMES),
                   *(PROJECT/'src/infra_calc/transport').glob('*.py'),
                   PROJECT/'src/infra_calc/topics/transport_sender.py',
                   PROJECT/'src/infra_calc/topics/transport_closed_loop.py'})


def rows(paths):
    return [dict(file=str(p.relative_to(PROJECT)), sha256=digest(p)) for p in paths]


def verify():
    manifest = json.loads((DIRECTORY/'manifest.json').read_text())
    expected = {'inputs': input_paths(), 'artifacts': [DIRECTORY/n for n in ARTIFACTS]}
    for key, paths in expected.items():
        actual = manifest.get(key, [])
        if len(actual) != len(paths) or {r['file'] for r in actual} != {str(p.relative_to(PROJECT)) for p in paths}:
            raise ValueError('Incomplete media-feedback figure manifest: '+key)
        for row in actual:
            if digest(PROJECT/row['file']) != row['sha256']:
                raise ValueError('Stale media-feedback figure: '+row['file'])
    return {'verified_figures': 2}


def extract():
    data = []
    for name in NAMES:
        result = json.loads((PROJECT/f'results/media-feedback-{name}.json').read_text())
        business = {b['kind']: b for b in result['businesses']}
        early = []
        for packet in result['transmissions']:
            if Fraction(packet['send_start']) > Fraction(1, 2):
                continue
            early.append({k: packet.get(k) for k in ('direction','pn','kind','message_id','send_start','send_end','received_at')})
        data.append(dict(name=name, image=business['image'], audio=business['tts'], screenshot=business['screenshot'],
                         summary=result['summary'], early_transmissions=early))
    return data


def render():
    from .reproduce import verify_results
    verify_results(include_figures=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    data = extract()
    labels = ['FIFO / immediate', 'FIFO / aggregate', 'Priority / immediate', 'Priority / aggregate']
    colors = ['#466da4', '#7b9cc7', '#b96738', '#d29c75']
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={'height_ratios':[1,1,1.45]})
    times = [float(Fraction(r['image']['complete_at'])) for r in data]
    axes[0].barh(labels, times, color=colors)
    for i,t in enumerate(times): axes[0].text(t+.08, i, f'{t:.6f} s', va='center', fontsize=9)
    axes[0].set_xlim(0,17)
    axes[0].set_xlabel('Full image delivered at client (s)')
    axes[0].set_title('A. Same declared workload; FIFO compute is unchanged', loc='left')
    missing = [float(Fraction(r['audio']['missing_audio_seconds'])) for r in data]
    played = [sum(float(Fraction(b['play_end'])-Fraction(b['play_start'])) for b in r['audio']['blocks'] if b['play_start'] is not None and b['play_end'] is not None) for r in data]
    axes[1].barh(labels, played, color='#48866e', label='Played')
    axes[1].barh(labels, missing, left=played, color='#ba5e62', label='Missing')
    for i,r in enumerate(data):
        first = r['audio']['first_play']
        text = 'No playback' if first is None else f'First play {float(Fraction(first)):.3f} s'
        axes[1].text(.165,i,text,va='center',fontsize=9)
    axes[1].set_xlim(0,.23); axes[1].legend(loc='lower right',fontsize=8)
    axes[1].set_xlabel('Declared 0.16 s audio: played and missing duration (s)')
    axes[1].set_title('B. Empty transport queues do not imply usable media',loc='left')
    ax=axes[2]
    for i,r in enumerate(data):
        for t in r['early_transmissions']:
            start,end=float(Fraction(t['send_start'])),float(Fraction(t['send_end']))
            lane=i+(0.12 if t['direction']=='up' else -0.12)
            if t['kind']=='ack':
                ax.plot(start,lane,'|',color='#555555',markersize=4,alpha=.45)
            else:
                ax.broken_barh([(start,end-start)],(lane-.075,.15),facecolors=colors[i])
    ax.set_yticks(range(4),labels);ax.set_xlim(0,.5)
    ax.set_xlabel('Actual serialization start/end, first 0.5 s (upper lane: download; lower: upload)')
    ax.set_title('C. Data/control service and actual ACK starts share the links',loc='left')
    ax.legend(handles=[Patch(color='#555555',label='ACK start marker')],fontsize=8,loc='upper right')
    for ax in axes: ax.invert_yaxis();ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Shared media with real feedback: 20 / 100 Mbps, 50 ms propagation each way',fontsize=12)
    fig.text(.08,.017,'Declared model time and media slots, not measured TCP/H3. Screenshot action is undelivered in all four cases.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96))
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):fig.savefig(DIRECTORY/f'figure.{suffix}',dpi=160)
    plt.close(fig)
    (DIRECTORY/'data.json').write_text(json.dumps({'scope':'Public finite simulation; not measured protocol comparison','cases':data},indent=2)+'\n')
    (DIRECTORY/'manifest.json').write_text(json.dumps({'inputs':rows(input_paths()),'artifacts':rows([DIRECTORY/n for n in ARTIFACTS])},indent=2)+'\n')
    return verify()
