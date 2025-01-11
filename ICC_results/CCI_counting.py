import pandas as pd
import pingouin as pg

# Example data (replace with your actual data)
data = {
    'Image': [1, 2, 3],
    'Ground Truth': [45, 32, 60],
    'Model 1 Prediction': [44, 31, 59],
    'Model 2 Prediction': [46, 33, 61],
    'Model 3 Prediction': [43, 34, 62]
}
df = pd.DataFrame(data)

# Melt the data for ICC calculation
df_melted = df.melt(id_vars='Image', var_name='Rater', value_name='Score')

# Calculate ICC
icc = pg.intraclass_corr(data=df_melted, targets='Image', raters='Rater', ratings='Score')

# Show ICC results
print(icc)
