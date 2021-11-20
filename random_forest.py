from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def build_random_forest(x_train, y_train):
    model = RandomForestClassifier(max_depth=6)
    model.fit(x_train, y_train)
    return model

def parse_first_tree(random_forest_model, parameter_names):
    text_representation = tree.export_text(random_forest_model.estimators_[0])

    with open("random_forest_first_tree.txt", "w+") as fout:
        fout.write(text_representation)

    fig = plt.figure(figsize=(50,50))
    _ = tree.plot_tree(random_forest_model.estimators_[0]
                        , feature_names=parameter_names
                        , filled=True)
    fig.savefig("random_forest_first_tree.png", bbox_inches="tight")