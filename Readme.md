# Batchpoeder Dashboard

A Streamlit dashboard for visualizing and analyzing batch production data from Excel files.

## Features

- Upload an Excel dataset (`.xlsx`) via the sidebar.
- View a quick overview of your uploaded data.
- Analyze success rates by Material Number.
- Interactive Altair chart for visualizing success rates.
- Select a Material Number to view detailed information.

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- pandas
- altair
- openpyxl

## Installation

```bash
pip install streamlit pandas altair openpyxl
```

## Usage

1. Place your Excel file in a known location.
2. Run the dashboard:

```bash
streamlit run main.py
```

3. Use the sidebar to upload your Excel file.
4. Explore your data and analysis in the main view.

## Notes

- The Excel file must contain at least the columns: `Material Number`, `Valuation`, and `Material Group`.
- The dashboard automatically handles missing or invalid numeric data.

## Example

![Dashboard Screenshot](screenshot.png)

---

© 2024 Batchpoeder Dashboard
