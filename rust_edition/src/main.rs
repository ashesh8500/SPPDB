use std::collections::HashMap;
use std::time::{Duration, UNIX_EPOCH};
use tokio_test;
use yahoo_finance_api as yahoo;

pub struct Portfolio {
    portfolio: HashMap<String, f64>,
    symbols: Vec<String>,
    cash_value: f64,
    liquid: f64,
    distribution: Option<HashMap<String, f64>>,
    price_distribution: Option<HashMap<String, f64>>,
}

impl Portfolio {
    pub fn new(portfolio: HashMap<String, f64>) -> Portfolio {
        let provider = yahoo::YahooConnector::new().unwrap();
        let symbols: Vec<String> = portfolio.keys().cloned().collect();
        let mut cash_value = 0.0;
        let mut distribution = HashMap::new();
        let mut price_distribution = HashMap::new();
        for ticker in symbols.iter() {
            let response =
                tokio_test::block_on(provider.get_quote_range(ticker, "1d", "1mo")).unwrap();
            cash_value += response.last_quote().unwrap().adjclose;
            price_distribution.insert(ticker.clone(), response.last_quote().unwrap().adjclose);
        }

        return Portfolio {
            portfolio,
            symbols,
            cash_value,
            liquid: 0.0,
            distribution: Some(distribution),
            price_distribution: Some(price_distribution),
        };
    }

    pub fn present_distribution(&self) -> HashMap<String, f64> {
        let total: f64 = self.portfolio.values().sum();
        self.portfolio
            .iter()
            .map(|(ticker, shares)| (ticker.clone(), shares / total))
            .collect()
    }
}

fn main() {
    let p = Portfolio::new(HashMap::from([
        ("AAPL".to_string(), 10.0),
        ("GOOGL".to_string(), 5.0),
        ("TSLA".to_string(), 5.0),
    ]));
    println!("{:?}", p.price_distribution);
}
