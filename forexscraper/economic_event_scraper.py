from bs4 import BeautifulSoup
from datetime import datetime

import pandas as pd
import re
import urllib.request


class ForexFactoryScraper():
    def __init__(self, month_select: str):
        self._url = 'https://www.forexfactory.com/calendar?month=' + month_select
        self._extracted_events = None

    def _extract_day(self, row_html):
        day_date = row_html.find(
            "td", {"class": "calendar__date"}).text.strip()
        return day_date[3:]

    def _extract_currency(self, row_html):
        return row_html.find("td", {"class": "calendar__currency"}).text.strip()

    def _extract_event(self, row_html):
        return row_html.find("td", {"class": "calendar__event"}).text.strip()

    def _extract_time(self, row_html):
        return row_html.find("td", {"class": "calendar__time"}).text.strip()

    def _extract_impact(self, row_html):
        return row_html.find("td", {"class": "calendar__impact"}).find("span").get('class')[1]
    
    def _extract_actual_forecast(self, row_html):
        actual = row_html.find("td", {"class": "calendar__actual"}).text.strip()
        forecast = row_html.find("td", {"class": "calendar__forecast"}).text.strip()
        better_or_worse_for_currency = row_html.find("td", {"class": "calendar__actual"}).find('span').get('class') if len(actual) > 0 else []
        better_or_worse_for_currency = better_or_worse_for_currency[0] if  len(better_or_worse_for_currency) > 0 else None
        return actual, forecast, better_or_worse_for_currency

    def _extract_html_data(self):

        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        response = opener.open(self._url)
        result = response.read().decode('utf-8', errors='replace')

        return BeautifulSoup(result, "html.parser")

    def _is_time(self, time_string):
        pattern = re.compile(r"^(\d{1,2}):(\d{2})(am|pm)$")
        return bool(pattern.match(time_string))
    
    def _get_compatible_datetime(self, date: str, time: str, year: str):
        if time == "All Day" or time[0].isalpha() or not self._is_time(time_string=time):
            time = '12:00am'
        date_string = year + " " + date + " " + time  # ex: "2024 Sep 2 5:20am"

        # Define the date format
        date_format = "%Y %b %d %I:%M%p"

        # Parse the date string
        date_object = datetime.strptime(date_string.strip(), date_format)
        
        # Format the date object to the desired output
        output_date_string = date_object.strftime("%Y-%m-%d %H:%M:%S")

        return output_date_string  # Output: 2024-09-02 05:20:00

    def extract_events(self, year: str = "2024"):
        parsed_html = self._extract_html_data()
        table = parsed_html.find_all("tr", class_="calendar__row")

        economic_events_list = []

        current_extracted_date = None
        current_time = None

        for row in table:
            if "calendar__row--day-breaker" in str(row.get('class')) or 'calendar__row--no-event' in str(row.get('class')):
                continue

            currency = self._extract_currency(row)

            # Sometimes there are no events. This can be checked via the currency
            if not currency:
                continue

            # Recurring day and date is blank
            if 'calendar__row--new-day' in str(row.get('class')):
                current_extracted_date = self._extract_day(
                    row) or current_extracted_date

            # Events at the same time is blank
            current_time = self._extract_time(row) or current_time

            event = self._extract_event(row)
            impact = self._extract_impact(row)

            comaptible_datetime = self._get_compatible_datetime(
                date=current_extracted_date, time=current_time, year=year)

            is_all_day = True if "All Day" == current_time else False

            actual, forecast, currency_strength = self._extract_actual_forecast(row)
            economic_events_list.append({
                'date': current_extracted_date,
                'time_minus_12hours': current_time,
                'currency': currency,
                'event': event,
                'impact': impact,
                'datetime': comaptible_datetime,
                "all_day": is_all_day,
                'actual': actual,
                'forecast': forecast,
                'currency_strength': currency_strength,
            })

        events_df = pd.DataFrame(economic_events_list)
        events_df['impact'] = events_df.impact.map({'icon--ff-impact-red': "High", 'icon--ff-impact-ora': "Medium",
                                                   'icon--ff-impact-yel': "Low", 'icon--ff-impact-gra': "None"}).fillna("None").astype(str)

        self._extracted_events = events_df

        return events_df

    def get_today_events(self):

        # Remove leading 0 from date.
        # If using non-windows, replace '#' with '-'
        current_date = datetime.now().strftime("%b %#d")
        extracted_events_copy = self._extracted_events.copy()

        filtered_events = extracted_events_copy[extracted_events_copy['date'] == current_date]
        filtered_events = filtered_events.groupby('currency')

        events_str = ""

        for currency, frame in filtered_events:

            events_str += f"{currency}\n\n"

            for index, row in frame.iterrows():
                events_str += f"{row['event']
                                 } at {row['time_minus_12hours']} (GMT-4: USA TIME).\n"
                events_str += f"Impact: {row['impact']}\n\n"

            events_str += "\n\n-----\n\n"

        return events_str

def get_all_events_from_year(year: str = "2024"):
    all_events = pd.DataFrame()
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for month in months:
        month = month + '.' + year
        events = ForexFactoryScraper(month_select=month).extract_events(year=year)
        all_events = pd.concat([all_events, events])
    return all_events

if __name__ == "__main__":

    # month = "dec" + '.' + "2024"
    # events = ForexFactoryScraper(month_select=month).extract_events(year="2024")
    
    events = get_all_events_from_year()
    events.to_csv("economic_calendar.csv", encoding='utf-8', index=False, header=True)

    print(events)
