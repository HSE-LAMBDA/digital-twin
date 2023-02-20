from digital_twin.data import POOL_DEPENDENT_VARS, POOL_INDEPENDENT_VARS

class Model:
    X_train, y_train, train_indices, train_shapes = Grouper().transform(
        train_df, dependent_vars=POOL_DEPENDENT_VARS, independent_vars=POOL_INDEPENDENT_VARS
    )
    X_test, y_test, test_indices, test_shapes = Grouper().transform(
        test_df, dependent_vars=POOL_DEPENDENT_VARS, independent_vars=POOL_INDEPENDENT_VARS
    )
    model = regressor(X_train, y_train, X_test, y_test)
    params = model.predict(X_test)
    params = pd.DataFrame(params, columns=y_test.columns)
    
def get_predictions(train_df, test_df, model_checkpoint_path):
    

    df = []
    for idx, (_, row) in zip(indices_test, params.iterrows()):
        gmm = GMM()
        gmm.init_from_params(
            weights=row.filter(regex="weight").values,
            means=row.filter(regex="mean").values.reshape(gmm.model.n_components, -1),
            precisions_cholesky=row.filter(regex="precision").values.reshape(
                gmm.model.n_components, -1
            ),
        )
        generated_samples = gmm.sample(len(idx))
        chunk_df = (
            kwargs.get("target_df")
            .loc[idx]
            .assign(
                **{
                    "generated_iops": generated_samples[:, 0],
                    "generated_lat": generated_samples[:, 1],
                }
            )
        )
        df.append(chunk_df)
    return pd.concat(df), indices_test