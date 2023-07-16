from bs4 import BeautifulSoup
import requests, csv
from random import randint
from time import sleep
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium import webdriver


def get_first_page_jobs(url, writer):
    #for each page, get the links of all the jobs
    try:
        source = requests.get(url)
        #check if the url is valid
        source.raise_for_status()
        
        soup = BeautifulSoup(source.text, 'lxml')

        links = soup.find('section', id='search-results-list').find_all('li')
        
        #for each job
        for temp in links:
            link = temp.a.get('href')
            link = 'https://www.capitalonecareers.com/' + link
            
            title = temp.a.find('h2').text
            location = temp.a.find('span', class_='job-location').text
            
            try:
                job_source = requests.get(link)
                job_source.raise_for_status()
                soup_temp = BeautifulSoup(job_source.text, 'html.parser')
                description = soup_temp.find('div', class_='ats-description').text
                # overview = soup_temp.find('div', id='page').find('main', id='content').find('section', class_='job-description global-vertical-top global-vertical-bottom')
                # description = overview.find('div', class_='job-description__wrap').find('div', class_='ats-description').text
            except Exception as e:
                print(e)
            
            writer.writerow([title, description, location, link])
        sleep(randint(2,10))
    except Exception as e:
            print(e)
            
def get_other_page_jobs(url, writer, links):
    links.extend([link.get_attribute('href') for link in driver.find_elements_by_css_selector('.hoverdetail a')])
    
    

          
csvfile = open('c1.csv', 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['Job Title', 'Description', 'Location', 'Link'])
          
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

url = 'https://www.capitalonecareers.com/search-jobs/United%20States/234/2/6252001/39x76/-98x5/50/2'
driver.get(url)
count = 0

while count == 0:
    get_first_page_jobs(url, writer)
    
    count += 1
    print("Page: ", count)
else:
    while True:
        try:
            count += 1
            print("Page: ", count)
            
            WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.CSS_SELECTOR, '.highslide-dimming.highslide-viewport-size')))
            next_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="pagination-bottom"]/div[2]/a[2]')))
            driver.execute_script("return arguments[0].scrollIntoView(true);", next_button)
            # Wait until the overlay is invisible

            next_button.click()
            
            print('clicked')

            # Wait for the elements to be loaded on the new page
            WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="search-results-list"]//li/a[@href]')))

            elems = driver.find_elements('xpath', '//*[@id="search-results-list"]//li/a[@href]')

            print('link found')
            jobcount = 1
            for i in range(len(elems)):
                try:
                    elem = driver.find_element('xpath', f'//*[@id="search-results-list"]//li[{i+1}]/a[@href]')
                    link = elem.get_attribute("href")
                    job_source = requests.get(link)
                    print(link)
                    job_source.raise_for_status()
                    job_soup = BeautifulSoup(job_source.text, 'lxml')
                    print('passed')
            
                    title = job_soup.find('h1').text
                    print('title' + str(jobcount)*10)
                    location = job_soup.find('span', class_="job-location").text
                    print('location' + str(jobcount)*10)
                    description = job_soup.find('div', class_='ats-description').text
                    print('description'+ str(jobcount)*10 + '\n')
                    
                    jobcount += 1
                except Exception as e:
                    print(e)
                
                writer.writerow([title, description, location, link])
                sleep(randint(2,10))
    
            print("Navigating to Next Page")
            
        except Exception or TimeoutException or WebDriverException as e:
            print(e)
            print("Last page reached")
            print(count)
            
            break
    
driver.quit()

csvfile.close()