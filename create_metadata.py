import json
import os
import re

for root, subfolders, files in os.walk('data'):
	for f in files:
		if f[-4:] == '.png' and 'full-250x250' in f:
			path = os.path.join(root, f)
			floats = re.findall(r'\d+\.\d+', path)
			t = float(floats[0])
			rho = float(floats[1])
			data.append({'path': path, 'label': [t, rho]})

json.dump(data, open('metadata/metadata.json', 'w'), indent=4, sort_keys=True)