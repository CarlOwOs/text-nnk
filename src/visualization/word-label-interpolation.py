import pandas as pd
import altair as alt
import ast
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='./data/test/data.csv')
parser.add_argument('--output_file', default='./visualization/test/visualization.html')

if __name__ == "__main__":
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_file, index_col=0)
    # if label == 2, make it 1
    df["label"] = df["label"].apply(lambda x: 1 if x == 2 else x)
    
    nnk_label = []
    for idx, row in df.iterrows():
        weight_values = ast.literal_eval(row.weight_values)
        indices = ast.literal_eval(row.indices)
        
        label_weight = 0
        
        for i in range(len(weight_values)):
            if weight_values[i] > 0:
                if df.iloc[[indices[i]]]["label"].item() == 0:
                    label_weight -= weight_values[i]    
                elif df.iloc[[indices[i]]]["label"].item() == 1:
                    label_weight += weight_values[i]

        nnk_label.append((row.label, int(label_weight > 0)))
        
    # Create a confusion matrix from the nnk_label list, where the first element is the true label and the second is the predicted label
    confusion_matrix = [[0, 0], [0, 0]]
    for item in nnk_label:
        confusion_matrix[item[0]][item[1]] += 1
    print(confusion_matrix)
    
    # Create a list of the confusion matrix values and the corresponding label
    data = []
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            data.append((i, j, confusion_matrix[i][j]))
    
    # Create a dataframe from the list
    df = pd.DataFrame(data, columns=["true_label", "predicted_label", "count"])
    print(df)
    
    # Create a chart from the dataframe
    ch = alt.Chart(df).mark_rect().encode(
        x=alt.X("true_label:N", title="True Label", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("predicted_label:N", title="Predicted Label", axis=alt.Axis(tickMinStep=1, grid=False)),
        color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="viridis"), legend=None),
        tooltip=["true_label", "predicted_label", "count"]
    ).properties(title="Confusion Matrix: NNK neighbors of each Label", width=300, height=300)
    
    # Save the chart to a file
    ch.save(args.output_file)