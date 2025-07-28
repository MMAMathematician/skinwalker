from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pyautogui
from PIL import ImageGrab, Image, ImageDraw
from datetime import datetime
import time
import random
import re
from openai import OpenAI

chars_per_minute = 320
chars_per_second = chars_per_minute / 60.0

message_polling_frequency = 2

def type(element, text):
    for char in text:
        time.sleep(random.uniform(0, 2 / chars_per_second))
        element.send_keys(char)
        
def human_click(x, y):
    time.sleep(random.uniform(0, 0.5))
    pyautogui.moveTo(x, y)
    pyautogui.click()
    
    
def human_drag(x1, y1, x2, y2):
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, 1, button='left')
        
def access_chat(wait, username):
    friend_link = wait.until(EC.element_to_be_clickable(
        (By.XPATH, f"//a[@aria-label='{username} (direct message)']")
    ))

    friend_link.click()
    
def get_history(wait, username, system_prompt):  
    msgs = wait.until(EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "div[class*='messageContent_']")
            ))

    history = [
        {
            'role': 'system',
            'content': system_prompt
        }
    ]
    
    for msg in msgs:
        try:
            header = msg.find_element(
                By.XPATH,
                "./preceding-sibling::h3[1]"
            )
            user_el = header.find_element(By.CSS_SELECTOR, "span[class*='username_']")
            
            user = user_el.text.strip()
            if user == username:
                role = 'user'
            else:
                role = 'assistant'
            
            content = msg.text.strip()

            history.append({
                "role": role,
                "content": content
            })
        except Exception:
            continue
        
    return history

def get_all_usernames(driver, wait):
    script = """
    return Array.from(
      document.querySelectorAll("a[aria-label$='(direct message)']")
    ).map(el =>
      el.getAttribute("aria-label").replace(/ \\(direct message\\)$/, "")
    );
    """
    return driver.execute_script(script)
    
def message_users(driver, wait, usernames, client, system_prompt):
    if not usernames:
        usernames = get_all_usernames(driver, wait)
  
    while True:
        for username in usernames:
            access_chat(wait, username)
            time.sleep(message_polling_frequency)
            
            history = get_history(wait, username, system_prompt)
            
            msg_box = wait.until(EC.visibility_of_element_located(
                (By.CSS_SELECTOR, "div[role='textbox'][contenteditable='true']")
            ))
            msg_box.click()
            
            if history[-1]['role'] == 'user':
                response = client.responses.create(
                model="gpt-4o-mini",
                input=history,
                )
                
                type(msg_box, response.output_text)
                msg_box.send_keys(Keys.RETURN)
            
def handle_captcha_instructions(instructions, abs_x, abs_y):
    lines = instructions.splitlines()
    
    command = None
    coords = []
    
    if 'CLICK' in lines[0]:
        command = 'CLICK'
        for line in lines[1:]:
            regex = re.search(r'([0-9]+),[ ]*([0-9]+)', line)
            if regex is None:
                continue
            coords.append((abs_x + int(regex.group(1)), abs_y + int(regex.group(2))))
    elif 'DRAG' in lines[0]:
        command = 'DRAG'
        for line in lines[1:]:
            regex = re.search(r'([0-9]+),[ ]*([0-9]+),[ ]*([0-9]+),[ ]*([0-9]+)', line)
            if regex is None:
                continue
            coords.append((abs_x + int(regex.group(1)), abs_y + int(regex.group(2))))
            coords.append((abs_x + int(regex.group(3)), abs_y + int(regex.group(4))))
    elif 'DONE' in lines[0]:
        command = 'DONE'
        pass
        
    return command, coords

def commit_instructions(command, coords):
    if command == 'CLICK':
        for coord in coords:
            human_click(coord[0], coord[1])
    elif command == 'DRAG':
        pairs = list(zip(coords[::2], coords[1::2]))
        for pair in pairs:
            human_drag(pair[0][0], pair[0][1], pair[1][0], pair[1][1])
        
        
            
def handle_captcha(driver, wait, client):
    iframe = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "iframe[src*='hcaptcha.com']")
    ))
    
    iframe_box = driver.execute_script("""
        const r = arguments[0].getBoundingClientRect();
        return { left: r.left + window.scrollX, top: r.top + window.scrollY };
    """, iframe)
    
    driver.switch_to.frame(iframe)
    
    checkbox = wait.until(EC.element_to_be_clickable(
        (By.ID, 'checkbox')
    ))
    checkbox.click()
    
    driver.switch_to.default_content()
    
    iframes = driver.find_elements(By.TAG_NAME, "iframe")
    
    driver.switch_to.frame(iframes[2])

    pane = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "div.challenge")
    ))
    
    pane_box = driver.execute_script("""
        const r = arguments[0].getBoundingClientRect();
        return { left: r.left, top: r.top, width: r.width, height: r.height };
    """, pane)
    
    abs_x = 0
    abs_y = 0
    bottom_x, bottom_y = pyautogui.size()
    
    time.sleep(2)
    
    image_prompt = "The image is a puzzle. Solve this puzzle by giving a set of instructions. First, print CLICK or DRAG depending on the type of puzzle, followed by a newline. Then, print a list of instructions (X, Y) (for CLICK) and (X1, Y1, X2, Y2) (for DRAG), where the X and Y's correspond to the pixel coordinates. Each instruction should be separated by a newline. If there are no instructions to print, then print DONE. The squares in the grid are 50 x 50 pixels. Images of the puzzle will be constantly sent until the puzzle is completed. Red crosses correspond to coordinates that have been given by you in your last set of instructions."
    
    history = [{
        'role': 'user',
        'content': [
            {'type': 'input_text', 'text': image_prompt}
        ]
    }]
    
    command = None
    coords = []
    
    while True:
        screenshot = ImageGrab.grab(bbox=(abs_x, abs_y, bottom_x, bottom_y))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot.save(f'screenshots/{timestamp}.png')

        img = Image.open(f'screenshots/{timestamp}.png')
        draw = ImageDraw.Draw(img)
        
        increment = 50
        
        # Draw vertical lines
        for x in range(increment, img.size[0], increment):
            draw.line((x, 0, x, img.size[1]), fill=128)
            
        # Draw horizontal lines
        for y in range(increment, img.size[1], increment):
            draw.line((0, y, img.size[0], y), fill=128)
            
        cross_size = 30
            
        # Draw red crosses
        for coord in coords:
            draw.line((coord[0] - cross_size / 2, coord[1] - cross_size / 2, coord[0] + cross_size / 2, coord[1] + cross_size / 2), fill=128, width=3)
            draw.line((coord[0] - cross_size / 2, coord[1] + cross_size / 2, coord[0] + cross_size / 2, coord[1] - cross_size / 2), fill=128, width=3)
            
        img.save(f'screenshots/{timestamp}.png')
        
        result = client.files.create(
            file=open(f'screenshots/{timestamp}.png', 'rb'),
            purpose='vision'
        )
        
        history += [{
            'role': 'user',
            'content': [
                {
                    'type': 'input_image',
                    'file_id': result.id
                }
            ]
        }]
            
        response = client.responses.create(
            model='o4-mini',
            input=history
        )
        
        print(response.output_text)
        
        new_command, new_coords = handle_captcha_instructions(response.output_text, abs_x, abs_y)
        commit_instructions(new_command, new_coords)
        
        if new_command == 'DONE':      
            try:
                verify = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "div.button-submit.button[aria-label='Verify Answers']")
                ))
                verify.click()
                break
            except Exception:
                pass
            
            try:
                next = wait.until(EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "div.button-submit.button[aria-label='Next Challenge']")
                ))
                next.click()
            except Exception:
                pass
        else:
            command = new_command
            coords = new_coords
            
        
    
        
def access_discord(login, password, gpt_token, system_prompt, usernames, automate_captcha, code_queue):
    client = OpenAI(
    api_key=gpt_token
    )
    
    driver = webdriver.Firefox()
    driver.maximize_window()
    driver.get("https://canary.discord.com/login")

    wait = WebDriverWait(driver, 5)

    try:
        continue_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[.//div[text()='Continue in Browser']]")
        ))
        continue_btn.click()
        
        wait.until(EC.staleness_of(continue_btn))
    except Exception:
        pass

    email_input = wait.until(EC.presence_of_element_located(
        (By.NAME, "email")
    ))
    email_input.clear()
    type(email_input, login)

    password_input = wait.until(EC.presence_of_element_located(
        (By.NAME, "password")
    ))
    password_input.clear()
    type(password_input, password)

    login_button = driver.find_element(
        By.CSS_SELECTOR,
        "button[type='submit']"
    )
    login_button.click()
    
    
    if automate_captcha:
        while True:
            try:
                handle_captcha(driver, wait, client)
                
            except Exception as e:
                print(e)
                break
    else:
        try:
            wait_longer = WebDriverWait(driver, 300)
            
            is_email = '@' in login
            
            if is_email:
                wait_longer.until(EC.url_contains('channels'))
            else:
                try:
                    wait.until(EC.url_contains('channels'))
                    
                except Exception:
                    boxes = wait_longer.until(EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "input.input__5ecaa.codeInput__584e1")
                    ))
                    
                    two_factor_code = code_queue.get(timeout=300)
                    
                    for box, digit in zip(boxes, two_factor_code):
                        box.click()
                        type(box, digit)
        except Exception as e:
            print(e)
    
    message_users(driver, wait, usernames, client, system_prompt)
    
    driver.close()