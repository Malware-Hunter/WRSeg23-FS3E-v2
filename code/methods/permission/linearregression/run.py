def add_arguments(parser):
    parser = parser.add_argument_group("Arguments for Linear Regression")
    parser.add_argument("--splits", type=int, help="Number of splits", required = False, default = 10)

def run(args,ds):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression



    dataset = pd.read_csv(ds)
    X = dataset.drop(args.class_column, axis=1)
    y = dataset[args.class_column]

    feature_names = np.array(X.columns.values.tolist())
    fold_count = [args.splits] * len(feature_names)
    fold_ft_num = []
    fold_ft_to_delete = []

    kf = KFold(n_splits=args.splits, shuffle=True, random_state=42)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        model = LinearRegression()
        model.fit(X_train, y_train)
        coef_in = model.coef_
        n1 = len(coef_in)
        features_list_to_delete = []

        for i in range(0, n1):
            if coef_in[i] < 0.1 and coef_in[i] > -0.1:
                features_list_to_delete.append(feature_names[i])
        fold_ft_num.append(len(features_list_to_delete))
        fold_ft_to_delete.append(features_list_to_delete)

        for ft in features_list_to_delete:
            index = list(feature_names).index(ft)
            fold_count[index]-=1

    max_value = None
    index = None

    for idx, num in enumerate(fold_ft_num):
        if max_value is None or num > max_value:
            max_value = num
            index = idx

    new_X = X.drop(fold_ft_to_delete[index], axis=1)
    df2 = pd.DataFrame(new_X)
    df2['class'] = y
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f'linearregression_{os.path.basename(ds)}')
    df2.to_csv(output_file, index=False)

    return True