import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np


st.title('Convert rows to matrix')

components.html(
    '''
    <p style="color:#FFF">Upload a CSV file to convert to a matrix format.</p>
    '''
)

uploaded_file = st.file_uploader("Choose your file")
if uploaded_file is not None:

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.dataframe(data=dataframe, width=None, height=None)

    # get header names
    headers = dataframe.columns.values.tolist()

    row_c = st.selectbox("Select Row Index", headers, index=0)
    col_c = st.selectbox("Select Column Header", headers, index=1)
    val_c = st.selectbox("Select Values", headers, index=2)

    if st.button('Create Matrix!'):
        # pivot the dataframe
        matrix_df = dataframe.reset_index().pivot_table(index=row_c, columns=col_c, values=val_c, aggfunc='mean')
        matrix_df = matrix_df.dropna(axis='columns')
        st.dataframe(data=matrix_df, width=None, height=None)

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(matrix_df)

        st.download_button(
            label="Download matrix as CSV",
            data=csv,
            file_name='matrix.csv',
            mime='text/csv',
        )