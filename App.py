import pandas as pd
import streamlit as st
import csv

def load_file(uploaded_file):
    try:
        # Try to sniff the delimiter
        uploaded_file.seek(0)
        sample = uploaded_file.read(1024).decode("utf-8")
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
        uploaded_file.seek(0)

        # Read the CSV
        df = pd.read_csv(uploaded_file, dtype=str, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
        return df
    except UnicodeDecodeError:
        # Handle non-UTF-8 files
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, dtype=str, encoding='ISO-8859-1', on_bad_lines='skip')
        return df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        return None

def main():
    st.title("Territory Distribution Tool")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            st.write("File loaded successfully!")
            st.write(df.head())
        else:
            st.error("Failed to process the file.")

if __name__ == "__main__":
    main()
