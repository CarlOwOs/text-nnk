import pandas as pd
import altair as alt

if __name__ == "__main__":
    #true_label  predicted_label   count
    #       0                0  193873
    #       0                1    8846
    #       1                0   18278
    #       1                1  102199
    # turn the data above into a dataframe
    df = pd.DataFrame([[0, 0, 193873], [0, 1, 8846], [1, 0, 18278], [1, 1, 102199]], columns=["true_label", "predicted_label", "count"])
    
    # Create a chart from the dataframe
    ch = alt.Chart(df).mark_rect().encode(
        x=alt.X("true_label:N", title="True Label", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y("predicted_label:N", title="Predicted Label", axis=alt.Axis(tickMinStep=1, grid=False)),
        color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="viridis"), legend=None),
        tooltip=["true_label", "predicted_label", "count"]
    ).properties(title="Confusion Matrix: NNK neighbors of each Label", width=300, height=300)

    # Save the chart to a file
    ch.save("./visualization/heatmap.html")