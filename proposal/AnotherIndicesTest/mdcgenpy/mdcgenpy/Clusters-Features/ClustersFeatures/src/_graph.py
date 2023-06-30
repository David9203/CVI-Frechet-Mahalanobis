from ClustersFeatures import settings
if settings.Activated_Graph:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go


    class __Graph(object):
        def graph_boxplots_distances_to_centroid(self, Cluster):
            """Shows a box plot of the distances between all elements and the centroid of given cluster.

            :param Cluster: Cluster centroid name to evaluate the elements distance with.
            :returns: Plotly figure instance.

            >>> CC.graph_boxplots_distances_to_centroid(CC.labels_clusters[0])
            """
            graph_colors = {Cluster: settings.discrete_colors[i] for i, Cluster in enumerate(self.labels_clusters)}
            if not (Cluster in self.labels_clusters):
                raise KeyError(
                    'A such cluster name "' + Cluster + '" isn\'t found in dataframe\'s clusters. Here are the available clusters : ' + str(
                        list(self.labels_clusters)))
            else:
                Result = pd.DataFrame(data=self.data_target.values, columns=['Cluster'])
                Result['Distance'] = self.data_every_element_distance_to_centroids[Cluster]
                fig = go.Figure()
                for Cluster_ in self.labels_clusters:
                    fig.add_trace(go.Box(y=Result.Distance[Result['Cluster'] == Cluster_], boxpoints='all', text="Distribution",
                                         name="Cluster " + str(Cluster_), marker_color=graph_colors[Cluster_]))
                fig.update_layout(title_text="Distance between all elements and the centroid of cluster " + str(Cluster))
                fig.show()

        def graph_confusion_hypersphere_evolution_for_linspace_radius(self, n_pts, proportion):
            """ Returns a Plotly animation with dataframes generated by the Confusion Hypersphere for different radius.

            This animation allows users to understand which clusters are more confused with each other. You can also interpret compactness as follows:
            The diagonal term (when proportion is True) that first reaches the value 1 corresponds to the most compact cluster in the dataset
            :param int n_pts: Number of points for the radius linspace.
            :param bool proportion: Put the value of proportion to Confusion Hypersphere arguments
            :returns: Plotly figure instance.

            >>> CC.graph_confusion_hypersphere_evolution_for_linspace_radius(50, True)
            """
            if not isinstance(n_pts, int):
                raise ValueError('n_pts is not an integer.')
            if not isinstance(proportion, bool):
                raise ValueError('proportion is not boolean.')

            max_radius=np.max([self.data_radius_selector_specific_cluster("max",Cluster) for Cluster in self.labels_clusters])
            radius_linspace=np.round(np.linspace(0, max_radius, 50),1)
            df_dict={radius: self.confusion_hypersphere_matrix(radius=radius, counting_type="including", proportion=proportion) for radius in radius_linspace}

            self.__graph_animated_dataframes(df_dict, title="Confusion Hypersphere : Evolution for a variable radius")

        def graph_reduction_2D(self, reduction_method):
            """Shows the 2D reduction graph with Plotly.

            :param str reduction_method: "UMAP" or "PCA"
            :returns: Plotly figure instance

            >>> CC.graph_reduction_2D("UMAP")

            """
            graph_colors = {Cluster: settings.discrete_colors[i] for i, Cluster in enumerate(self.labels_clusters)}
            if not reduction_method in ['PCA','UMAP']:
                raise ValueError('reduction_method is not in ' + str(['PCA','UMAP']))

            if reduction_method == "UMAP":
                Mat=self.utils_UMAP()
            else:
                Mat=self.utils_PCA(2)
            data=pd.DataFrame(Mat)
            data['Cluster'] = self.data_target
            fig=go.Figure()
            for Cluster in self.labels_clusters:
                fig.add_trace(go.Scatter(x=data[data['Cluster'] == Cluster][data.columns[0]],y=data[data['Cluster'] == Cluster][data.columns[1]], name="Cluster " + str(Cluster),mode="markers",marker_color=graph_colors[Cluster],opacity=0.90))
            fig.update_layout(title="2D "+ reduction_method +" Projection")
            fig.show()

        def graph_PCA_3D(self):
            """Shows the 3D PCA reduction graph with Plotly.

            :returns: Plotly figure instance

            >>> CC.graph_PCA_3D()

            """
            graph_colors = {Cluster: settings.discrete_colors[i] for i, Cluster in enumerate(self.labels_clusters)}
            Mat=self.utils_PCA(3)
            data=pd.DataFrame(Mat)
            data['Cluster'] = self.data_target
            fig=go.Figure()
            for Cluster in self.labels_clusters:
                fig.add_trace(go.Scatter3d(x=data[data['Cluster'] == Cluster][data.columns[0]], y=data[data['Cluster'] == Cluster][data.columns[1]], z=data[data['Cluster'] == Cluster][data.columns[2]],name="Cluster " + str(Cluster),marker_color=graph_colors[Cluster],mode='markers'))
            fig.update_layout(title="3D PCA Projection")
            fig.show()




        def graph_reduction_density_3D(self,percentile,**args):
            """Shows the result of 3D PCA density estimation with Plotly.

            :param int percentile: Sets the minimum density contour to select as a percentile of the current density distribution.
            :param list cluster=: A list of clusters to estimate density.
            :returns: Plotly figure instance

            >>> CC.graph_reduction_density_3D(99,cluster=CC.labels_clusters[:2])

            >>> CC.graph_reduction_density_3D(99,cluster=CC.labels_clusters[0])

            >>> CC.graph_reduction_density_3D(99)
            """
            unpacked_dict= self.density_projection_3D(percentile, return_grid=True, return_clusters_density=True)

            each_cluster_density_save=unpacked_dict['Clusters Density']
            A = unpacked_dict['A-Grid']
            X = unpacked_dict['3D Grid']['X']
            Y = unpacked_dict['3D Grid']['Y']
            Z = unpacked_dict['3D Grid']['Z']

            fig = go.Figure()
            try:
                clusters=args['cluster']
                if (isinstance(clusters,str) or isinstance(clusters,float) or isinstance(clusters,int)):
                    if not clusters in self.labels_clusters:
                        raise ValueError(str(clusters) +' is not in ' +str(self.labels_clusters))
                    else:
                        fig.add_trace(go.Volume(
                            x=X.flatten(),
                            y=Y.flatten(),
                            z=Z.flatten(),
                            name="Cluster" + str(clusters),
                            value=each_cluster_density_save[clusters].flatten(),
                            isomin=np.percentile(each_cluster_density_save[clusters], percentile),
                            isomax=np.max(each_cluster_density_save[clusters]),
                            surface_count=20,
                        ))
                elif isinstance(clusters,list) or isinstance(clusters,np.ndarray):
                    for Cluster in clusters:
                        if not Cluster in self.labels_clusters:
                            raise ValueError(str(Cluster) + ' is not in ' + str(self.labels_clusters))
                    if len(clusters)>2:
                        raise ValueError('Computing more than 2 clusters is disabled for density 3D')
                    else:
                        list_colorscale = ["Blues", "Reds"]
                        for i, Cluster in enumerate(clusters):
                            fig.add_trace(go.Volume(
                                x=X.flatten(),
                                y=Y.flatten(),
                                z=Z.flatten(),
                                name="Cluster" + str(Cluster),
                                value=each_cluster_density_save[Cluster].flatten(),
                                isomin=np.percentile(each_cluster_density_save[Cluster], percentile),
                                isomax=np.max(each_cluster_density_save[Cluster]),
                                surface_count=20,
                                colorscale=list_colorscale[i]))
                            fig.update_traces(showlegend=False)
                else:
                    raise ValueError('Unknown type for clusters argument')
            except KeyError:
                fig.add_trace(go.Volume(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    name="All clusters",
                    value=A.flatten(),
                    isomin=np.percentile(A, percentile),
                    isomax=np.max(A),
                    surface_count=20))

            fig.update_layout(scene_xaxis_showticklabels=False,
                              scene_yaxis_showticklabels=False,
                              scene_zaxis_showticklabels=False)

            fig.show()

        def graph_reduction_density_2D(self, reduction_method,percentile, graph):
            """Shows the result of 2D PCA density estimation with Plotly.

            :param str reduction_method: "UMAP" or "PCA". Reduces the total dimension of the dataframe to 2.
            :param int percentile: Sets the minimum density contour to select as a percentile of the current density distribution.
            :param str graph: "interactive" or "contour". Shows different ways to visualize the density.

            :returns: Plotly figure instance.
            """
            graph_colors = {Cluster: settings.discrete_colors[i] for i, Cluster in enumerate(self.labels_clusters)}

            if not reduction_method in ['PCA','UMAP']:
                raise ValueError('reduction_method is not in ' + str(['PCA','UMAP']))

            if not graph in ['contour','interactive']:
                raise ValueError('graph argument is not in ' + str(['contour','interactive']))

            unpacked_dict = self.density_projection_2D(reduction_method, percentile, return_data=True,
                                                             return_clusters_density=True)
            Zi = unpacked_dict['Z-Grid']
            data = unpacked_dict['2D PCA Data']
            R = unpacked_dict['Clusters Density']

            Zi[Zi<np.percentile(Zi, percentile)] = 0
            if graph == "interactive":
                fig = go.Figure(
                    go.Contour(
                        x=Zi.index.values,
                        y=Zi.columns.values,
                        z=Zi,
                        contours_coloring='heatmap',
                        colorscale='Greys',
                        opacity=0.75,
                        name="Density"
                    ))
                fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                })
                centroids = {}
                clusters_circle = []
                for Cluster in self.labels_clusters:
                    data_cluster = data[self.data_target == Cluster]
                    centroids[Cluster] = data_cluster.mean()
                    xcenter = centroids[Cluster].values[0]
                    ycenter = centroids[Cluster].values[1]
                    dx = np.percentile(data_cluster[data_cluster.columns[0]], 75) - np.percentile(
                        data_cluster[data_cluster.columns[0]], 25)
                    dy = np.percentile(data_cluster[data_cluster.columns[1]], 75) - np.percentile(
                        data_cluster[data_cluster.columns[1]], 25)
                    clusters_circle.append(dict(type="circle", fillcolor=graph_colors[Cluster], opacity=0.25,
                                                line=dict(color="#000000", width=1), xref="x", yref="y", text=Cluster,
                                                x0=xcenter - dx, y0=ycenter - dx, x1=xcenter + dx, y1=ycenter + dx))

                fig.add_trace(go.Scatter(x=pd.DataFrame(centroids).loc[data.columns[0]], name="Centroid",
                                         y=pd.DataFrame(centroids).loc[data.columns[1]], text=["Centroid of cluster " + str(cl) for cl in self.labels_clusters],
                                         marker=dict(color=[settings.discrete_colors[Cluster] for Cluster in self.labels_clusters]),
                                         mode='markers'))


                fig.add_trace(go.Scatter(x=data[data.columns[0]].sample(frac=0.2, random_state=1), name="Point",
                                       y=data[data.columns[1]].sample(frac=0.2, random_state=1), mode="markers", marker_color=self.data_target.sample(frac=0.2, random_state=1).apply(lambda x: settings.discrete_colors[x]),
                                         marker=dict(size=2.5), opacity=0.70,text=["Point of cluster " + str(cl) for cl in self.data_target.sample(frac=0.2, random_state=1)]))

                button = dict(method='relayout',
                              label="Show clusters",
                              args=["shapes", []],
                              args2=["shapes", clusters_circle])
                um = dict(buttons=[button], showactive=False, type='buttons', y=1.12, x=0.20)
                fig.update_layout(showlegend=False, updatemenus=[um], title="2D Density Projection")

                fig.show()

            elif graph == "contour":
                Z = np.zeros(Zi.shape)
                contours_ = []
                for i, Cluster in enumerate(self.labels_clusters):
                    Z += R[Cluster]
                    z = np.round(1 * (R[Cluster] > np.percentile(R[Cluster], percentile)) * R[Cluster], 1)
                    contours_.append(go.Contour(
                        x=Zi.index.values,
                        y=Zi.columns.values,
                        z=z,
                        name="Cluster " + str(Cluster),
                        hoverinfo='skip',
                        line=dict(color=graph_colors[Cluster]),
                        contours=dict(type="constraint")
                    ))
                fig = go.Figure(data=contours_)
                fig.update_layout(title="2D Density Projection")
                fig.show()

        def graph_projection_2D(self, feature1, feature2):
            """A simple 2D projection on two given features with Plotly.

            :param feature1: The first dataframe columns to project
            :param feature2: The second dataframe columns to projectv

            :returns: Plotly figure instance.

            """

            if not(feature1 in self.data_features.columns) or not(feature2 in self.data_features.columns):
              raise ValueError('Specified feature is not in the available features.')
            else:

                graph_colors = {Cluster: settings.discrete_colors[i] for i, Cluster in enumerate(self.labels_clusters)}

                fig = go.Figure()
                clusters_circle = []

                for Cluster in self.labels_clusters:
                    xcenter, ycenter = self.data_clusters[Cluster][[feature1, feature2]].mean().values
                    dx = np.percentile(self.data_clusters[Cluster][feature1], 75) - np.percentile(
                        self.data_clusters[Cluster][feature1], 25)
                    dy = np.percentile(self.data_clusters[Cluster][feature2], 75) - np.percentile(
                        self.data_clusters[Cluster][feature2], 25)

                    fig.add_trace(
                        go.Scatter(x=self.data_clusters[Cluster][feature1], y=self.data_clusters[Cluster][feature2], mode="markers", name="Cluster "+str(Cluster),
                                   marker_color=self.data_target[self.data_clusters[Cluster].index].apply(lambda x: graph_colors[x])))
                    clusters_circle.append(dict(type="circle", fillcolor=graph_colors[Cluster], opacity=0.35,
                                                line=dict(color="#000000", width=1), xref="x", yref="y",name = "Cluster " + str(Cluster),
                                                x0=xcenter - dx, y0=ycenter - dx, x1=xcenter + dx, y1=ycenter + dx))

                button = dict(method='relayout',
                              label="Show clusters",
                              args=["shapes", []],
                              args2=["shapes", clusters_circle])
                um = dict(buttons=[button], showactive=False, type='buttons', y=1.12, x=0.20)
                fig.update_layout(title=f"2D Projection on feature {str(feature1)} and feature {str(feature2)}", showlegend=True,
                                  updatemenus=[um])
                fig.show()




        def __graph_animated_dataframes(self, dict_df, **args):
            """
            Animate a dict of dataframe with plotly sliders and buttons. The key is the name of the dataframe and the item is a Pandas Dataframe
            """
            if isinstance(dict_df, dict):
                list_df = list(dict_df.values())
                # Check if there is no values in the list that are not pandas dataframes
                if np.count_nonzero([not (isinstance(x, pd.DataFrame)) for x in list_df]) != 0:
                    raise TypeError('The given dict is not full of Pandas dataframes.')
            else:
                raise TypeError('The given dict has not a dict type.')

            try:
                fill_color = args['fill_color']
            except KeyError:
                fill_color = "#2980b9"
            try:
                width = args['width']
                height = args['height']
            except KeyError:
                width = 1000
                height = 550
            try:
                title = args['title']
            except KeyError:
                title = ""

            # Create figure
            fig = go.Figure(frames=[go.Frame(data=go.Table(
                header=dict(values=df.columns, fill_color=fill_color, align="center"),
                cells=dict(values=np.round([df[i].values for i in df.columns],2), align="center")
            ), name=list(dict_df.keys())[k]) for k, df in enumerate(list_df)])

            # Add the first dataframe
            fig.add_trace(go.Table(
                header=dict(
                    values=list(list_df[0].columns),
                    font=dict(size=10),
                    fill_color=fill_color,
                    align="center"
                ),
                cells=dict(
                    values=np.round([list_df[0][i].values for i in list_df[0].columns],2),
                    align="center")
            ))

            # Create function to get the time between frames
            frame_args = lambda duration: {"frame": {"duration": duration}, "mode": "immediate", "fromcurrent": True,
                                           "transition": {"duration": duration, "easing": "linear"}}

            # Add the slider
            sliders = [dict(
                active=0,
                len=0.75,
                x=0.14,
                y=-0.05,
                currentvalue={"prefix": "Value "},
                pad={"t": 1},
                steps=[{"args": [[f.name], frame_args(0)], "label": list(dict_df.keys())[k], "method": "animate"} for k, f in
                       enumerate(fig.frames)]
            )]
            # Add the buttons
            fig.update_layout(
                title=title,
                width=width,
                height=height,
                sliders=sliders,
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(500)],
                                "label": "&#9654;",  # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;",  # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "right",
                        "pad": {"r": 10, "t": 30},
                        "type": "buttons",
                        "x": 0.12,
                        "y": 0.07,
                    }
                ]
            )

            fig.show()


        def __graph_multivariate_plot(self, df, columns, **args):
            """
            Plot severals columns in the dataframe with Plotly. This allows to add a "target=" argument in order to visualize the data with its associated cluster
            """
            xaxis_set = "date"
            if isinstance(df, pd.DataFrame):
                df.columns = df.columns.astype(str)
                if not (isinstance(df.index, pd.DatetimeIndex)):
                    xaxis_set = "-"
            else:
                raise TypeError('Given dataframe isn\'t a Pandas dataframe')

            if type(columns) == "str":
                columns = [columns]
            for col in columns:
                if col not in df.columns:
                    raise AttributeError('One of the given columns is not in dataframe\'s columns.')

            try:
                target = args['target']
                if len(target) != len(df):
                    raise ValueError('The target hasn\'t the same lenght as the given dataframe.')
                if isinstance(target, pd.Series):
                    target = target.to_list()
            except:
                target = None

            try:
                title = args['title']
            except:
                title = "Multivariate Plot"

            try:
                cluster_opacity = args['cluster_opacity']
            except:
                cluster_opacity = 0.3

            df = df.sort_index()
            cluster_colors = px.colors.qualitative.Safe
            curves_colors = px.colors.qualitative.Dark24
            yaxis_name_converter = lambda x: str(x + 1) if (x != 0) else ""

            fig = go.Figure()
            for i, col in enumerate(columns):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, text=df[col], marker_color=curves_colors[i],
                                         yaxis="y" + yaxis_name_converter(i)))
            fig.update_traces(hoverinfo="name+x+text", line={"width": 0.5}, marker={"size": 8}, mode="lines+markers",
                              showlegend=False, )

            df_index = df.index.insert(len(df.index), '')

            if not (target is None):
                fig.update_layout(shapes=[
                    dict(fillcolor=cluster_colors[Cluster], opacity=cluster_opacity, line={"width": 0}, type="rect",
                         layer="below", x0=df_index[i], x1=df_index[i + 1], y0=0, y1=1, yref="paper") for i, Cluster in
                    enumerate(target)])

            xaxis = dict(autorange=True, type=xaxis_set, rangeslider=dict(autorange=True))
            fig.update_layout(xaxis=xaxis)

            fig.update_layout({"yaxis" + yaxis_name_converter(i): dict(
                anchor="x",
                range=[df[col].min(), df[col].max()],
                autorange=True,
                domain=[(i) / len(columns), (i + 1) / len(columns)],
                linecolor=curves_colors[i],
                mirror=True,
                showline=True,
                side="right",
                tickfont={"color": curves_colors[i]},
                tickmode="auto",
                ticks="",
                title=col[0:8],
                titlefont={"color": curves_colors[i]},
                type="linear",
                zeroline=False
            ) for i, col in enumerate(columns)})

            fig.update_layout(dragmode="zoom", hovermode="x", legend=dict(traceorder="reversed"), height=500, width=1450,
                              title=title, template="plotly_white", margin=dict(t=60, b=20))

            fig.show()


else:
    class __Graph:
        def __init(self):
            pass