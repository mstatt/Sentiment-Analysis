# Sentiment-Analysis

Basic sentiment analysis useing Vader.


1) Reads the files from "froprocessing/" individually into a string.
2) Removes punctuation
3) Runs the sentiment analysis loading the dictionary
4) Converts the Sentiment dictionary into a dataframe
5) Writes the output to an html file.

The results will look like the example below:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Files Analyzed</th>
      <th>Positive Score</th>
      <th>Neutral Score</th>
      <th>Negative Score</th>
      <th>Overall Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>filename.txt</td>
      <td>0.164</td>
      <td>0.807</td>
      <td>0.029</td>
      <td>0.135</td>
    </tr>
    <tr>
      <th>2</th>
      <td>filename2.txt</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
