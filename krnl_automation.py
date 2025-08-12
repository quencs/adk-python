import re
import time
import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# キー形式チェック
def check_krnl_key(key: str):
    pattern = re.compile(r'^[0-9a-fA-F]{4}(-[0-9a-fA-F]{4}){3}$')
    return bool(pattern.match(key))

# KRNLキー自動取得
def auto_get_key_full():
    print("\n[+] KRNLキー取得自動化を開始します...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://krnl.gg/getkey")

    wait = WebDriverWait(driver, 60)
    krnl_key = None

    try:
        # 「次へ」クリック
        next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '次へ')]")))
        time.sleep(1)
        next_btn.click()
        print("[+] '次へ'ボタンをクリックしました")

        # 「続行」クリック
        time.sleep(5)
        continue_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '続行')]")))
        continue_btn.click()
        print("[+] '続行'ボタンをクリックしました")

        # キー取得
        key_element = wait.until(EC.presence_of_element_located((By.XPATH, "//code")))
        krnl_key = key_element.text.strip()
        pyperclip.copy(krnl_key)
        print(f"[+] 取得したキー: {krnl_key} （クリップボードにコピーしました）")

    except Exception as e:
        print("[-] 自動化中にエラー:", e)
    finally:
        driver.quit()

    return krnl_key

# KRNLにキーを入力してInject
def inject_krnl(key):
    print("\n[+] KRNLにキーを入力してInjectします...")
    time.sleep(5)  # ユーザーにKRNL画面をアクティブにしてもらう時間

    # キー入力欄クリック（座標は環境に合わせて調整）
    pyautogui.click(500, 400)  # 仮の座標
    time.sleep(0.5)

    # キー貼り付け
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)

    # Submitクリック（座標も調整必要）
    pyautogui.click(600, 500)
    time.sleep(1)

    print("[+] キー送信完了、Injectが開始されます。")

if __name__ == "__main__":
    user_key = input("既にKRNLキーを持っていますか？（なければEnter）: ").strip()

    if user_key and check_krnl_key(user_key):
        print("\nこのキーは形式上OKです。Injectを開始します。")
        inject_krnl(user_key)
    else:
        print("\nキーが無効 or 入力なしのため、自動取得します。")
        key = auto_get_key_full()
        if key:
            inject_krnl(key)