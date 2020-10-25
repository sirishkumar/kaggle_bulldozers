import fastbook
from fastbook import *

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG

path = Path('./data')


dep_var = 'SalePrice'
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df_test = pd.read_csv(path/'Test.csv', low_memory=False)

# We need to optimize the algorithm for RMSLE
df[dep_var] = np.log(df[dep_var])

# Handling Dates
def handle_dates(df):
    df = add_datepart(df, 'saledate')

def transform_data(df):
    cond = (df.saleYear <= 2011) & (df.saleMonth <= 10)
    train_idx = np.where(cond)[0]
    valid_idx = np.where(~cond)[0]
    splits = (list(train_idx), list(valid_idx))    

    cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
    procs = [Categorify, FillMissing]
    to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
    return to


def get_dtreeviz(m, df):
    xs, y = df.xs, df.y
    samp_idx = np.random.permutation(len(y))[:500]
    viz = dtreeviz(m, xs.iloc[samp_idx], y.iloc[samp_idx], xs.columns, dep_var,
                   fontname='DejaVu Sans', scale=1.6, label_fontsize=10,
                   orientation='LR')
    file_name = '/tmp/dtree.svg'
    viz.save(file_name)
    return file_name


def r_mse(pred, y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)

def write_metrics(file_name, train_rmse, valid_rmse):

    with open(file_name, 'w') as fp:
        t_rmse = "%2.1f" % train_rmse
        v_rmse = "%2.1f" % valid_rmse
        fp.write("Training RMSE: %s\n" % t_rmse)
        fp.write("Validation RMSE: %s\n" % v_rmse)

    with open("metrics.json", 'w') as outfile:
        json.dump({"training_rmse": t_rmse, "valid_rmse": v_rmse}, outfile)

handle_dates(df)
handle_dates(df_test)

# Transform data
to = transform_data(df)

# Train DT
xs, y = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y
m = DecisionTreeRegressor(max_leaf_nodes=10)
m.fit(xs, y)
train_rmse = m_rmse(m, xs, y)
valid_rmse = m_rmse(m, valid_xs, valid_y)

write_metrics("metrics.txt", train_rmse, valid_rmse)

draw_tree(m, xs, size=7, leaves_parallel=True, precision=2)
get_dtreeviz(m, to.train)



