import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO


def read_csv_data(file_path):

    data = pd.read_csv(file_path)
    return data

class DataPreparation:

    def __init__(self, data):

        self.data = data
    
    def process_data(self):
       
        prep_data = self.data.iloc[:,2:] #we are removing the first 3 columns from the left because we are not supposed to use them
        mag_columns = ['MAG_u','MAG_g','MAG_r','MAG_i','MAG_z']
        condition = (prep_data[mag_columns].isin([99.0, -99.0]))
        prep_data[mag_columns] = prep_data[mag_columns].mask(condition, None)  #remove missing values labeled with 99.0 and -99.0 and replace with None

        prep_data_profile = prep_data.iloc[:,1:]
        return prep_data, prep_data_profile

class DataProfile:
   
    def __init__(self, data):

        self.data = data

    def collect_stats(self):
       
        column_names = self.data.columns.tolist()
        statistics_dict = {}

        for column in self.data.columns:
            
            mean = self.data[column].mean()
            median = self.data[column].median()
            max = self.data[column].max()
            min = self.data[column].min()
            missing_values = self.data[column].isnull().sum()

            statistics_dict[column] = {

                'mean': mean,
                'median':median,
                'max':max,
                'min':min,
                'missing_values': missing_values

            }

        return statistics_dict

    def generate_histogram(self, bins = 10, output_path = 'histogram'):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for column in self.data.columns:

            data_without_null = self.data[column].dropna()

            plt.hist(data_without_null, bins = bins, edgecolor = 'black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')

            plt.title(f'Histogram for {column}')

            output_file = os.path.join(output_path, f'{column}_histogram.png')
            plt.savefig(output_file)

            plt.close()

            print(f'Histogram saved at: {output_file}')

    def find_missing_values(self):

        missing_values_dict = {}
        for column in self.data.columns:
            missing_rows = self.data[self.data[column].isnull()].index.tolist()
            if missing_rows:
                missing_values_dict[column] = missing_rows
            else:
                missing_values_dict[column] = []

        return missing_values_dict

    def generate_pdf(self, output_path='data_profile.pdf'):
        
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        try:

            for column, stats in self.collect_stats().items():

                pdf.drawString(15, 750, f"Feature: {column}")
                pdf.drawString(15, 730, f"Mean: {stats['mean']}")
                pdf.drawString(15, 710, f"Median: {stats['median']}")
                pdf.drawString(15, 690, f"Max: {stats['max']}")
                pdf.drawString(15, 670, f"Min: {stats['min']}")
                pdf.drawString(15, 650, f"Missing Values: {stats['missing_values']}")

                histogram_path = os.path.join('histogram', f'{column}_histogram.png')
                pdf.drawInlineImage(histogram_path, 10, 200, width=480, height=360)

                pdf.showPage()
        finally:
            pdf.save()

        buffer.seek(0)

        with open(output_path, 'wb') as f:
            f.write(buffer.read())

        print(f'Combined PDF report saved at: {output_path}')


        



