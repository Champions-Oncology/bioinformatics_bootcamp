import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
from umap import UMAP

st.title('Simple UMAP')
st.markdown('Uniform Manifold Approximation and Projection for Dimension Reduction') 

components.html(
    '''
    <p style="color:#FFF">Upload RNAseq data in matrix format to explore principal components</p>
    '''
)


st.markdown('Upload a CSV file of your sample identifiers in one column and some variable of interest in another column')
sample_file = st.file_uploader("Choose your sample mapping file")
if sample_file is not None:
    sample_df = pd.read_csv(sample_file)
    st.dataframe(data=sample_file, width=None, height=None)

st.markdown('Upload a CSV matric of your data') 
st.markdown('Row index is the sample identifier and column index is the variable of interest')
data_file = st.file_uploader("Choose your matrix values file")
if data_file is not None:
    df = pd.read_csv(sample_file)
    st.dataframe(data=df, width=None, height=None)


st.markdown('### Explore UMAP Hyperparameters') 
st.markdown('https://umap-learn.readthedocs.io/en/latest/') 

st.markdown('Number of Neighbors') 
n_neighbors = st.slider('Number of Neighbors', min_value=1, max_value=100, value=15)

if st.button('Plot UMAP'):
    # Configure UMAP hyperparameters
    reducer = UMAP(n_neighbors=n_neighbors, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
                n_components=2, # default 2, The dimension of the space to embed into.
                metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
                n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
                learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
                init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
                spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
                low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
                set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
                local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
                repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
                negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
                transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
                a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
                random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
                metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
                angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
                target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
                #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
                #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
                transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
                verbose=False, # default False, Controls verbosity of logging.
                unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
                )

    # Fit and transform the data
    X_trans = reducer.fit_transform(data_file.to_numpy()) #_df.to_numpy()

    umap_df = pd.DataFrame(X_trans, columns=['UMAP1','UMAP2'])
    # Add index to values_df
    new_df = data_file.reset_index()
    # Add column matching index
    finalDf = pd.concat([umap_df, new_df[['model']]], axis = 1)
    types_df = sample_df.reset_index()
    finalUMAPDf = finalDf.set_index('model').join(types_df.set_index('model'))
    print(finalUMAPDf)


    # Plot
    plotly_df = finalUMAPDf.reset_index()
    fig = px.scatter(plotly_df[['UMAP1', 'UMAP2', 'tumor_type', 'model']], 
                    x="UMAP1", y="UMAP2", color="tumor_type", symbol="tumor_type", 
                    hover_data=['model'], title="UMAP")
    fig.show()