### Integrate HTML With Flask
### HTTP verb GET And POST

##Jinja2 template engine
'''
{%...%} conditions,for loops
{{    }} expressions to print output
{#....#} this is for comments
'''
from flask import Flask,redirect,url_for,render_template,request
app=Flask(__name__)
@app.route('/')
def welcome():
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)