import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from typing import Dict, List, Optional


class SECAnalyzer:
    def __init__(self):
        self.base_url = "https://www.sec.gov/search-fillings/"
        self.headers = {
            "User-Agent": "Victory Ochei victoryocheii90@email.com"  # Replace with your details as per SEC requirements
        }

    def get_company_filings(self, ticker: str) -> pd.DataFrame:
        """
        Fetch all 10-K filings for a given company ticker.
        """
        try:
            # Get the company's CIK number
            search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany&type=10-K"
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract document links and dates
            docs = []
            for row in soup.find_all('tr'):
                cols = row.find_all('td')
                if len(cols) >= 4 and '10-K' in cols[0].text:
                    docs.append({
                        'date': cols[3].text,
                        'link': cols[1].find('a')['href'] if cols[1].find('a') else None
                    })

            return pd.DataFrame(docs)

        except Exception as e:
            print(f"Error fetching filings: {str(e)}")
            return pd.DataFrame()

    def extract_financial_data(self, filing_text: str) -> Dict:
        """
        Extract key financial metrics from 10-K filing text.
        """
        financial_data = {
            'revenue': None,
            'net_income': None,
            'total_assets': None,
            'total_liabilities': None,
            'cash_flow_operations': None
        }

        # Regular expressions for finding financial metrics
        patterns = {
            'revenue': r'Total revenue[s]?\s*[\$]?([\d,]+)',
            'net_income': r'Net income[s]?\s*[\$]?([\d,]+)',
            'total_assets': r'Total assets\s*[\$]?([\d,]+)',
            'total_liabilities': r'Total liabilities\s*[\$]?([\d,]+)',
            'cash_flow_operations': r'Net cash (?:provided by|used in) operating activities\s*[\$]?([\d,]+)'
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, filing_text)
            if match:
                value = match.group(1).replace(',', '')
                financial_data[metric] = float(value)

        return financial_data

    def analyze_trends(self, financial_history: List[Dict]) -> Dict:
        """
        Analyze financial trends over time.
        """
        df = pd.DataFrame(financial_history)

        analysis = {
            'revenue_growth': [],
            'profit_margin': [],
            'debt_ratio': [],
            'year_over_year_changes': {}
        }

        # Calculate year-over-year growth rates
        for column in df.columns:
            if column != 'date':
                df[f'{column}_growth'] = df[column].pct_change() * 100
                analysis['year_over_year_changes'][column] = df[f'{column}_growth'].tolist()

        # Calculate key ratios
        if 'net_income' in df.columns and 'revenue' in df.columns:
            analysis['profit_margin'] = (df['net_income'] / df['revenue'] * 100).tolist()

        if 'total_liabilities' in df.columns and 'total_assets' in df.columns:
            analysis['debt_ratio'] = (df['total_liabilities'] / df['total_assets'] * 100).tolist()

        return analysis

    def generate_report(self, ticker: str) -> Dict:
        """
        Generate a comprehensive financial analysis report for a company.
        """
        filings = self.get_company_filings(ticker)
        financial_history = []

        for _, filing in filings.iterrows():
            # Respect SEC rate limits
            time.sleep(0.1)

            try:
                response = requests.get(filing['link'], headers=self.headers)
                financial_data = self.extract_financial_data(response.text)
                financial_data['date'] = filing['date']
                financial_history.append(financial_data)
            except Exception as e:
                print(f"Error processing filing from {filing['date']}: {str(e)}")
                continue

        analysis = self.analyze_trends(financial_history)

        return {
            'company': ticker,
            'financial_history': financial_history,
            'trend_analysis': analysis
        }


def main():
    # Example usage
    analyzer = SECAnalyzer()

    # List of companies to analyze
    companies = ['AAPL', 'MSFT', 'GOOGL']

    for ticker in companies:
        print(f"\nAnalyzing {ticker}...")
        report = analyzer.generate_report(ticker)

        # Display key metrics
        print("\nKey Financial Metrics:")
        for period in report['financial_history']:
            print(f"\nPeriod: {period['date']}")
            for metric, value in period.items():
                if metric != 'date':
                    print(f"{metric}: ${value:,.2f}")

        # Display trend analysis
        print("\nTrend Analysis:")
        for metric, changes in report['trend_analysis']['year_over_year_changes'].items():
            print(f"\n{metric} year-over-year changes:")
            for change in changes:
                if change is not None:
                    print(f"{change:.2f}%")


if __name__ == "__main__":
    main()