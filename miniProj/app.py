from flask import Flask, render_template
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

app = Flask(__name__)

@app.route('/')
def index():
    # Load and process data
    df = pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')
    pr_num = df.select_dtypes(include=['int64', 'float64'])
    x = pr_num.iloc[:, 0:3].values
    y = pr_num.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    # Train model and predict
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Calculate R² score
    r2 = r2_score(y_test, y_pred) * 100

    # Create plots
    hist_plot = create_hist_plot(df['Profit'])
    scatter_plot = create_scatter_plot(df)
    bar_plot = create_bar_plot(y_test, y_pred)
    reg_plot = create_reg_plot(y_test, y_pred)

    return render_template('index.html', hist_plot=hist_plot, scatter_plot=scatter_plot, bar_plot=bar_plot, reg_plot=reg_plot, r2=r2)

def create_hist_plot(data):
    plt.figure()
    sns.histplot(data)
    plt.title('Profit Distribution')
    return save_plot_to_base64()

def create_scatter_plot(df):
    plt.figure()
    sns.scatterplot(x='R&D Spend', y='Profit', data=df, hue='State')
    plt.title('R&D Spend vs Profit')
    return save_plot_to_base64()

def create_bar_plot(y_test, y_pred):
    plt.figure()
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1.plot(kind='bar', figsize=[10, 5])
    plt.title('Actual vs Predicted Profit')
    return save_plot_to_base64()

def create_reg_plot(y_test, y_pred):
    plt.figure()
    df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    sns.regplot(x='Actual', y='Predicted', data=df1, color='green')
    plt.title('Regression Plot')
    return save_plot_to_base64()

def save_plot_to_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return img_str

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=3000,debug=True)
