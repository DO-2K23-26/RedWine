# wine

## Setup

```bash
pip install -r requirements.txt
```

## Run

### For the app
```bash
streamlit run app.py
```

### For the notebook
```bash
jupyter notebook
```

For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)