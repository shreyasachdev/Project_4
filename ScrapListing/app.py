  
from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
from scrapy_listing import scrape_listing
from model import Test_model


app = Flask(__name__)
scrapa = scrape_listing()
cars_attributes = {} 

# route
@app.route("/")
def index():
    print(cars_attributes)
    return render_template("index.html", mars = cars_attributes)

@app.route("/scrape")
def scrape():
    global cars_attributes
    cars_attributes = (scrapa.scrape_ad())
    cars_attributes['predict_model'] =  Test_model()
    return redirect("http://localhost:5000/")

if __name__ == "__main__":
    app.run(debug=True)