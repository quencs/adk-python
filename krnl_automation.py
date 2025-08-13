import re
import time
import subprocess
import pyautogui
import pyperclip
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Linux用のKRNLパス（実際のパスに変更してください）
KRNL_PATH = "/path/to/krnl"  # ここに実際のKRNL実行ファイルパスを入れてください

def check_krnl_key(key: str):
    pattern = re.compile(r'^[0-9a-fA-F]{4}(-[0-9a-fA-F]{4}){3}$')
    return bool(pattern.match(key))

def auto_get_key_full():
    print("\n[+] KRNLキー取得自動化を開始します...")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://krnl.gg/getkey")

    wait = WebDriverWait(driver, 60)
    krnl_key = None

    try:
        next_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '次へ')]")))
        time.sleep(1)
        next_btn.click()
        print("[+] '次へ'ボタンをクリックしました")

        time.sleep(5)
        continue_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '続行')]")))
        continue_btn.click()
        print("[+] '続行'ボタンをクリックしました")

        key_element = wait.until(EC.presence_of_element_located((By.XPATH, "//code")))
        krnl_key = key_element.text.strip()
        pyperclip.copy(krnl_key)
        print(f"[+] 取得したキー: {krnl_key} （クリップボードにコピーしました）")

    except Exception as e:
        print("[-] 自動化中にエラー:", e)
    finally:
        driver.quit()

    return krnl_key

def focus_krnl_window():
    """Linux用のウィンドウフォーカス関数"""
    try:
        # xdotoolを使用してKRNLウィンドウをアクティブにする
        subprocess.run(["xdotool", "search", "--name", "KRNL", "windowactivate"], check=True)
        time.sleep(1)
        print("[+] KRNLウィンドウをアクティブにしました")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[-] xdotoolが見つからないか、KRNLウィンドウが見つかりません。")
        print("[-] 手動でKRNLウィンドウをアクティブにしてください。")
        time.sleep(5)  # ユーザーに手動でアクティブにしてもらう時間
        return True

def inject_krnl(key):
    print("\n[+] KRNLにキーを入力してInjectします...")
    if not focus_krnl_window():
        return
    
    # キー入力欄クリック（位置は環境に合わせて調整）
    pyautogui.click(500, 400)
    time.sleep(0.5)
    
    # キー貼り付け
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    
    # Injectボタンクリック（位置は調整が必要）
    pyautogui.click(600, 500)
    time.sleep(1)
    
    print("[+] Inject処理完了しました。")

if __name__ == "__main__":
    print("[+] KRNLを起動します...")
    try:
        subprocess.Popen([KRNL_PATH])
        time.sleep(10)  # 起動待機（PC環境によって調整してください）
    except FileNotFoundError:
        print("[-] KRNL実行ファイルが見つかりません。")
        print("[-] KRNL_PATHを正しいパスに設定してください。")
        print("[-] 手動でKRNLを起動してください。")
        time.sleep(5)

    user_key = input("既にKRNLキーを持っていますか？（なければEnter）: ").strip()

    if user_key and check_krnl_key(user_key):
        print("\nこのキーは形式上OKです。Injectを開始します。")
        inject_krnl(user_key)
    else:
        print("\nキーが無効 or 入力なしのため、自動取得します。")
        key = auto_get_key_full()
        if key:
            inject_krnl(key)