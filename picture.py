import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.size'] = 20                  

root_dir2 = r'E:\SSN\ssn\data\period2\final600.xlsx'
myAnnotion2 = pd.read_excel(root_dir2).values
chlorophyll2 = myAnnotion2[:, 7].astype(np.float32)
nitrogen2 = myAnnotion2[:, 8].astype(np.float32)

root_dir3 = r'E:\SSN\ssn\data\period3\final600.xlsx'
myAnnotion3 = pd.read_excel(root_dir3).values
chlorophyll3 = myAnnotion3[:, 7].astype(np.float32)
nitrogen3 = myAnnotion3[:, 8].astype(np.float32)

root_dir4 = r'E:\SSN\ssn\data\period4\final600.xlsx'
myAnnotion4 = pd.read_excel(root_dir4).values
chlorophyll4 = myAnnotion4[:, 7].astype(np.float32)
nitrogen4 = myAnnotion4[:, 8].astype(np.float32)

root_dir5 = r'E:\SSN\ssn\data\period5\final600.xlsx'
myAnnotion5 = pd.read_excel(root_dir5).values
chlorophyll5 = myAnnotion5[:, 7].astype(np.float32)
nitrogen5 = myAnnotion5[:, 8].astype(np.float32)

periods = ['Stem elongation', 'Heading', 'Grain filling', 'Maturity']
data = {
    'Period': np.repeat(periods, 600),
    'Chlorophyll': np.concatenate([chlorophyll2, chlorophyll3, chlorophyll4, chlorophyll5]),
    'Nitrogen': np.concatenate([nitrogen2, nitrogen3, nitrogen4, nitrogen5])
}
df = pd.DataFrame(data)

palette = {'Stem elongation': '#66C2A5', 'Heading': '#FC8D62',
           'Grain filling': '#8DA0CB', 'Maturity': '#E78AC3'}

plt.figure(5)
for period in periods:
    sns.histplot(df[df['Period']==period]['Chlorophyll'], 
                 color=palette[period], 
                 kde=True, 
                 alpha=0.5, 
                 label=period,
                 bins=15)
plt.title('Chlorophyll Content Distribution')
plt.xlabel('Chlorophyll (SPAD)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(6)
for period in periods:
    sns.histplot(df[df['Period']==period]['Nitrogen'], 
                 color=palette[period], 
                 kde=True, 
                 alpha=0.5, 
                 label=period,
                 bins=15)
plt.title('Nitrogen Content Distribution')
plt.xlabel('Nitrogen (mg/g)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()



models = ['V6', 'VT', 'R3', 'R6', 'V6 & VT & R3 & R6']
metrics = ['R²', 'RPD', 'RMSE', 'MAE']

data = {
    'R²':   [0.866,0.881,0.899,0.849,0.951],
    'RMSE': [12.48,11.77,11.37,14.18,8.68],
    'MAE':  [2.73,2.90,3.15,2.57,4.50],
    'RPD':  [8.56,7.31,7.17,9.81,5.28]
}

plt.figure(figsize=(16, 8))  

colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D']

vertical_offsets = {
    'R²': 0.02,
    'RPD': 0.05,
    'RMSE': -0.02,
    'MAE': -0.05
}

for i, metric in enumerate(metrics):
    line = plt.plot(models, data[metric],
             color=colors[i],
             linestyle=line_styles[i % len(line_styles)],
             marker=markers[i],
             markersize=8,
             linewidth=2,
             alpha=0.8,
             label=metric)
    
    for j, value in enumerate(data[metric]):
        offset = 0.15 if (i == 0 or i == 1) else -0.25 
        if metric == 'R²':
            text = f'{value:.3f}'
        else:
            text = f'{value:.2f}'
            
        plt.text(j, data[metric][j] + vertical_offsets[metric] + offset, 
                text,
                color=colors[i],
                ha='center',
                va='bottom' if vertical_offsets[metric] > 0 else 'top',
                fontsize=20,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.xticks(ha='center')
plt.ylabel('Metric Value', fontsize=25)

best_r2_idx = np.argmax(data['R²'])
plt.gca().get_xticklabels()[best_r2_idx].set_color('red')
plt.gca().get_xticklabels()[best_r2_idx].set_fontweight('bold')

best_value = data['R²'][best_r2_idx]
plt.text(best_r2_idx, best_value - 0.7, '★', ha='center', va='bottom', color='red', fontsize=16, fontfamily='DejaVu Sans')

plt.text(0.02, 0.95, '← R², RPD: Higher better', transform=plt.gca().transAxes, 
         color='green', fontsize=16, bbox=dict(facecolor='white', alpha=0.7))
plt.text(0.02, 0.90, '← RMSE, MAE: Lower better', transform=plt.gca().transAxes, 
         color='red', fontsize=16, bbox=dict(facecolor='white', alpha=0.7))

plt.legend(bbox_to_anchor=(0.88, 0.77), framealpha=1, fontsize=20)
plt.grid(True, linestyle='--', alpha=0.3)

y_min = min([min(values) for values in data.values()])
y_max = max([max(values) for values in data.values()])
plt.ylim(y_min - 0.15*(y_max-y_min), y_max + 0.05*(y_max-y_min))

plt.tight_layout()
plt.show()   

