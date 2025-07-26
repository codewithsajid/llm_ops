# google_ai_scraper.py
import os
import rich
import time
import random
import tempfile
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def find_chrome_executable() -> str | None:
    """Finds a usable Chrome or Chromium binary, prioritizing a portable installation."""
    portable_path = os.path.expanduser("~/chrome/chrome")
    if os.path.isfile(portable_path) and os.access(portable_path, os.X_OK):
        rich.print(f"[green]✅ Found portable Chrome at:[/] {portable_path}")
        return portable_path
    
    import shutil
    for command in ["google-chrome", "chrome", "chromium-browser", "chromium"]:
        if path := shutil.which(command):
            rich.print(f"[green]✅ Found system Chrome at:[/] {path}")
            return path
            
    rich.print("[red]❌ No Chrome or Chromium executable found.[/]")
    return None

def scrape_google_ai_overview(question: str, timeout: int = 15, proxy: str = None) -> str | None:
    """
    Scrapes Google's AI-powered search results using advanced anti-detection techniques.
    """
    driver = None
    browser_executable_path = find_chrome_executable()
    if not browser_executable_path:
        return None

    try:
        rich.print("[cyan]Initializing stealth Chrome driver...[/]")
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
        
        if proxy:
            options.add_argument(f'--proxy-server={proxy}')
            rich.print(f"[cyan]Using proxy:[/] {proxy.split('@')[1]}")

        driver = uc.Chrome(options=options, browser_executable_path=browser_executable_path, use_subprocess=True)
        rich.print("[green]✅ Driver initialized.[/]")

        driver.get("https://www.google.com/")
        time.sleep(random.uniform(1.5, 3))

        search_box = driver.find_element(By.NAME, "q")
        for char in question:
            search_box.send_keys(char)
            time.sleep(random.uniform(0.04, 0.1))
        search_box.submit()
        
        rich.print(f"[cyan]Searching for:[/] '{question}'")

        # More robust selectors for the AI overview
        ai_selectors = [
            "div[data-g-bofa*='true']",           # A common container for AI answers
            "div[data-sgrd*='true']",             # The original selector, used as a fallback
            "div.exp-outline"                    # Selector for experimental outlines
        ]

        wait = WebDriverWait(driver, timeout)
        for i, selector in enumerate(ai_selectors):
            try:
                ai_overview_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                rich.print(f"[green]✅ AI-powered overview found with selector {i+1}.[/]")
                return ai_overview_element.text.strip()
            except Exception:
                continue

        # If no AI overview is found, raise an exception to be caught below
        raise TimeoutError("AI overview element not found with any selector.")

    except Exception:
        rich.print(f"[red]❌ An error occurred during scraping.[/]")
        if driver and "unusual traffic" in driver.page_source.lower():
            rich.print("[yellow]   -> Google's bot detection was triggered. This is common on server IPs.[/]")
        else:
            rich.print("[yellow]   -> The AI overview was not generated for this query or the page timed out.[/]")
        return None
        
    finally:
        if driver:
            driver.quit()
            rich.print("[yellow]🧹 Driver has been closed.[/]")

if __name__ == "__main__":
    test_question = "What are the latest developments in reinforcement learning?"
    
    scraped_text = scrape_google_ai_overview(test_question)
    
    print("\n" + "=" * 60)
    if scraped_text:
        rich.print("[bold green]✅ Scraping Successful! AI Overview:[/]")
        rich.print(f"\n[italic]{scraped_text}[/italic]")
    else:
        rich.print("[bold red]❌ Scraping Failed.[/]")
    print("=" * 60)



