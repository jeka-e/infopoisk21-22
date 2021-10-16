

import pandas as pd
import streamlit as st

"""
# My first app
Here's our first attempt at using data to create a table:
"""

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

option = st.selectbox('How would you like to be contacted?',
                      ('Email', 'Home phone', 'Mobile phone'))

