import re
content = open('outputs/presentation/thesis_presentation.html', encoding='utf-8').read()
print('File length chars:', len(content))
keys = ['title-slide', 'beh-legibility', 'beh-predictability',
        'beh-safety', 'beh-grounding', 'combined', 'vlm-disc',
        'conclusions', 'pipeline', 'obs-space']
for k in keys:
    print(f'Has {k}: {k in content}')
imgs = re.findall(r'data:image/png;base64,', content)
print('Embedded images:', len(imgs))
