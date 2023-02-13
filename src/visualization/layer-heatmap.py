import pandas as pd
import altair as alt

if __name__ == "__main__": 
    df = pd.read_csv('./data/results-layers.csv')
    
    # colnames is Index(['Layer', 'ABSum', 'ABAvg', 'MinS', 'AvgMinS', 'Avg'], dtype='object')
    # make it so that there are 3 columns: Layer, Metric, Score
    df = pd.melt(df, id_vars=['Layer'], var_name='Metric', value_name='Score')

    base = alt.Chart(df).encode(
        x=alt.X('Metric:N', title='Metric', scale=alt.Scale(paddingInner=0)),
        y=alt.Y('Layer:N', title='Layer', scale=alt.Scale(paddingInner=0), sort=['Layer-0', 'Layer-1', 'Layer-2', 'Layer-3', 'Layer-4', 'Layer-5', 'Layer-6', 'Layer-7', 'Layer-8', 'Layer-9', 'Layer-10', 'Layer-11', 'Layer-12', 'Layer-Avg']),
    ).properties(
        width=500,
        height=500,
    )

    heatmap = base.mark_rect().encode(
        color = alt.Color('Score:Q', title='Spearman Correlation', scale=alt.Scale(scheme='viridis')),
    )
    
    text = base.mark_text(baseline='middle').encode(
        text=alt.Text('Score:Q', format='.2f'),
        color=alt.condition(
            alt.datum.Score > 50,
            alt.value('black'),
            alt.value('white')
        )
    )

    (heatmap + text).save('visualization/results-layers.html')