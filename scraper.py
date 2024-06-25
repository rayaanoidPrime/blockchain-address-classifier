import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import chromedriver_autoinstaller
import csv
import time
import argparse

def setup_driver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chromedriver_autoinstaller.install()
    return webdriver.Chrome(options=chrome_options)

# scrape 10 times every 100 seconds
def scrape_sol(iterations=10, delay=100):
    driver = setup_driver()
    transactions_url = "https://solanabeach.io/transactions"
    base_url = "https://solanabeach.io"
    csv_filename = "datasets/sol.csv"

    for i in range(iterations):
        driver.get(transactions_url)
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "table"))
            )
        except TimeoutException:
            print("Main table not found within the given time")
            continue

        time.sleep(2)  # Small delay to ensure the page is fully loaded

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        table_rows = soup.find_all('tr')

        hrefs = []
        for row in table_rows:
            tds = row.find_all('td')
            for td in tds:
                a_tags = td.find_all('a')
                for a in a_tags:
                    href = a.get('href')
                    if href and "block" not in href:
                        hrefs.append(href)

        accounts = []
        for href in hrefs:
            driver.get(base_url + href)
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "table"))
                )
            except TimeoutException:
                print("Table not found within the given time")
                continue

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            table_rows = soup.find_all('tr')

            for row in table_rows:
                tds = row.find_all('td')
                if len(tds) != 0:
                    td = tds[0]
                    span = td.find('span')
                    if span:
                        div = span.find('div')
                        if div:
                            divs = div.find_all('div')
                            for div_ in divs:
                                a = div_.find('a')
                                if a:
                                    href = a.get('href')
                                    if href:
                                        try:
                                            account = href.split('address/')[1]
                                            accounts.append(account)
                                        except IndexError:
                                            print(f"Skipping invalid href: {href}")
                                            continue

        print(f"Iteration {i+1} accounts:", accounts)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["Account", "Blockchain"])
            for account in accounts:
                writer.writerow([account, "solana"])

        time.sleep(delay)

    driver.quit()
    print(f"Data has been appended to {csv_filename} for all iterations")

def scrape_oklink(transactions_url, output_csv, chain, driver):
    all_links = []
    driver.get(transactions_url)
    driver.save_screenshot("screenshot.png")

    def extract_links_from_table():
        max_attempts = 3
        attempts = 0
        while attempts < max_attempts:
            try:
                table_body = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "okui-table-content"))
                )
                table_rows = table_body.find_elements(By.TAG_NAME, "tr")

                for row in table_rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 7:  # Ensure we have enough cells
                        from_cell = cells[4]  # "From" column
                        to_cell = cells[6]    # "To" column

                        for cell in [from_cell, to_cell]:
                            links = cell.find_elements(By.TAG_NAME, "a")
                            for link in links:
                                href = link.get_attribute('href')
                                if href and "address/" in href:
                                    all_links.append(href.split("address/")[1])

                return True
            except StaleElementReferenceException:
                attempts += 1
                if attempts == max_attempts:
                    print("Max attempts reached. Moving to next page.")
                    return False
                print(f"Stale element encountered. Retrying... (Attempt {attempts})")
                time.sleep(2)  # Wait before retrying
            except TimeoutException:
                print("Table body not found within the given time")
                return False

    # Scrape transaction table details for 400 pages
    for page in range(400):
        if not extract_links_from_table():
            continue  # Move to next page instead of breaking

        print(f"Processed page {page + 1}")

        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "li.okui-pagination-next.okui-pagination-simple-icon"))
            )
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(2)  # Wait for page to load
        except Exception as e:
            print(f"Error clicking next button on page {page + 1}: {str(e)}")
            break

    # Create new csv file and write all the addresses in two columns "Account", "Blockchain"
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Account', 'Blockchain'])
        for link in all_links:
            writer.writerow([link, chain])

def main():
    parser = argparse.ArgumentParser(description="Scrape blockchain transactions")
    parser.add_argument("action", choices=["sol", "oklink"], help="Choose which blockchain to scrape")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for Solana scraping")
    parser.add_argument("--delay", type=int, default=100, help="Delay between iterations in seconds")
    
    args = parser.parse_args()

    if args.action == "sol":
        scrape_sol(args.iterations, args.delay)
    elif args.action == "oklink":
        driver = setup_driver()
        chains = [
            "bsc",      # BNB Chain
            "polygon",
            "avax",     # Avalanche
            "trx",      # TRON
            # "btc",    # Bitcoin  - enable if oklink scrapped data is needed otherwise publicly abailable dataset is in datasets folder
            # "eth"     # Ethereum
        ]
        for chain in chains:
            transactions_url = f"https://www.oklink.com/{chain}/tx-list/large"
            output_csv = f"datasets/{chain}_out.csv"
            scrape_oklink(transactions_url, output_csv, chain, driver)
            time.sleep(2)
            print(f"Data has been appended to {output_csv} for {chain}")
        driver.quit()
        print("Data has been appended for all chains")

if __name__ == "__main__":
    main()