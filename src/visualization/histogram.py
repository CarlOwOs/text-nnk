import pandas as pd
import altair as alt
import ast
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='./data/test/data.csv')
parser.add_argument('--output_file', default='./visualization/test/neighbor_chart.html')

if __name__ == "__main__":
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_file, index_col=0)
    
    data = [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]
     
    for idx, row in df.iterrows():
        neg_count = 0
        neu_count = 0
        pos_count = 0
        
        row_weight_values = ast.literal_eval(row.weight_values)
        row_indices = ast.literal_eval(row.indices)
        
        for i in range(len(row_weight_values)):
            if row_weight_values[i] > 0:
                if df.iloc[[row_indices[i]]]["label"].item() == 0:
                    neg_count += 1
                elif df.iloc[[row_indices[i]]]["label"].item() == 1:
                    neu_count += 1
                elif df.iloc[[row_indices[i]]]["label"].item() == 2:
                    pos_count += 1
                    
        if neg_count not in data[row.label][0]:
            data[row.label][0][neg_count] = 1
        else:
            data[row.label][0][neg_count] += 1
        
        if neu_count not in data[row.label][1]:
            data[row.label][1][neu_count] = 1
        else:
            data[row.label][1][neu_count] += 1
            
        if pos_count not in data[row.label][2]:
            data[row.label][2][pos_count] = 1
        else:
            data[row.label][2][pos_count] += 1
            
    color_palette = ["orange", "blue", "green"]
    
    neg_neg = [(item[0], item[1], 0) for item in list(data[0][0].items())]
    neg_neu = [(item[0], item[1], 1) for item in list(data[0][1].items())]
    neg_pos = [(item[0], item[1], 2) for item in list(data[0][2].items())]
    neg_data = pd.DataFrame(neg_neg + neg_neu + neg_pos, columns=["neighbors", "count", "label"])
    
    ch_neg = alt.Chart(neg_data).mark_area(opacity=0.5, interpolate="step").encode(
        x=alt.X("neighbors:Q", title="Number of neighbors", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("count:Q", title="Count", stack=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("label:N", title="Label", scale=alt.Scale(domain=[0, 1, 2], range=color_palette), legend=None),
    ).properties(title="Negative Sentiment: Number of neighbors per label", width=300, height=300)
    
    neu_neg = [(item[0], item[1], 0) for item in list(data[1][0].items())]
    neu_neu = [(item[0], item[1], 1) for item in list(data[1][1].items())]
    neu_pos = [(item[0], item[1], 2) for item in list(data[1][2].items())]
    neu_data = pd.DataFrame(neu_neg + neu_neu + neu_pos, columns=["neighbors", "count", "label"])
    
    ch_neu = alt.Chart(neu_data).mark_area(opacity=0.5, interpolate="step").encode(
        x=alt.X("neighbors:Q", title="Number of neighbors", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("count:Q", title="Count", stack=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("label:N", title="Label", scale=alt.Scale(domain=[0, 1, 2], range=color_palette), legend=None),
    ).properties(title="Neutral Sentiment: Number of neighbors per label", width=300, height=300)
    
    pos_neg = [(item[0], item[1], 0) for item in list(data[2][0].items())]
    pos_neu = [(item[0], item[1], 1) for item in list(data[2][1].items())]
    pos_pos = [(item[0], item[1], 2) for item in list(data[2][2].items())]
    pos_data = pd.DataFrame(pos_neg + pos_neu + pos_pos, columns=["neighbors", "count", "label"])
    
    ch_pos = alt.Chart(pos_data).mark_area(opacity=0.5, interpolate="step").encode(
        x=alt.X("neighbors:Q", title="Number of neighbors", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("count:Q", title="Count", stack=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color("label:N", title="Label", scale=alt.Scale(domain=[0, 1, 2], range=color_palette), legend=None),
    ).properties(title="Positive Sentiment: Number of neighbors per label", width=300, height=300)
    
    # legend, where 0 is negative, 1 is neutral, 2 is positive
    ch_legend = alt.Chart(pd.DataFrame([
        {"label": 0, "sentiment": "Negative"}, 
        {"label": 1, "sentiment": "Neutral"}, 
        {"label": 2, "sentiment": "Positive"}])).mark_rect(opacity=0.5).encode(
        y = alt.Y("sentiment:N", title="Sentiment", axis=alt.Axis(tickMinStep=1, grid=False)),
        color = alt.Color("label:N", title="Label", scale=alt.Scale(domain=[0, 1, 2], range=color_palette)),
    ).properties(title="Legend")
    
    (ch_neg | ch_neu | ch_pos | ch_legend).save(args.output_file)

        