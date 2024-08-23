import logging
import sys
import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import io


# Function to download a file from S3
def download_file_from_s3(s3_file_path, bucket, s3client):
    logging.info(f"Downloading file from S3: {s3_file_path}")
    logging.info(f"Downloading file from S3: {bucket}")
    try:
        with io.BytesIO() as data:
            s3client.download_fileobj(bucket, s3_file_path, data)
            data.seek(0)
            df = pd.read_csv(data)
        return df
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        raise

# Function to upload a file to S3
def upload_file_to_s3(bucket_name, s3_key, local_path, s3client):
    try:
        with open(local_path, 'rb') as data:
            s3client.upload_fileobj(data, bucket_name, s3_key)
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise

def run():
    """Main ETL script definition.

    :return: None
    """
    # S3 client initialization
    s3client = boto3.client('s3')   

    try:
        s3_file_path = "ExampleData_Launch.csv"
        bucket = 'conveyor-powertrain-data'
        output_key = 'output.csv'
        output_image_key = 'fft_plot.png'

        # Download the file from S3
        df = download_file_from_s3(s3_file_path, bucket, s3client)

        # Process the data
        time_sec = df['Time_sec']
        speed_kph = df['Speed_kph']

        # Plotting the original data
        plt.plot(df['Time_sec'], df['Speed_kph'])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (kph)')
        plt.title('Vehicle speed vs Time')
        plt.grid()
        plt.savefig('speed_vs_time_plot.png')
        # Calculate the sampling rate
        dt = np.mean(np.diff(time_sec))  # Time step (assume uniform sampling)
        fs = 1 / dt  # Sampling frequency

        # Define the high-pass filter
        def highpass_filter(data, cutoff_freq, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            filtered_data = filtfilt(b, a, data)
            return filtered_data

        # Apply the high-pass filter
        cutoff_frequency = 2  # Cutoff frequency (in Hz)
        filtered_speed_kph = highpass_filter(speed_kph, cutoff_frequency, fs)

        # Plot the filtered signal
        plt.plot(time_sec, filtered_speed_kph)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Filtered vehicle speed (kph)')
        plt.title('High-pass Filtered Signal')
        plt.grid()

        # Perform FFT
        N = len(speed_kph)  # Number of samples
        yf = np.fft.fft(speed_kph)  # Compute FFT
        xf = np.fft.fftfreq(N, dt)  # Frequency bins

        # Only keep the positive frequencies
        positive_freq_indices = np.where(xf >= 0)
        xf = xf[positive_freq_indices]
        yf = np.abs(yf[positive_freq_indices])  # Magnitude of FFT

        # Plot the FFT result
        plt.plot(xf, yf)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT of Vehicle speed')
        plt.grid()
        plt.ylim(0, 300)
        plt.savefig('fft_plot.png')
        plt.show()

        # Upload the FFT plot to the output S3 bucket
        upload_file_to_s3(bucket, output_image_key, 'fft_plot.png', s3client)

        # Detect peaks in the FFT result
        peaks, _ = find_peaks(yf, height=100)  # Adjust the height threshold as needed

        # Print peak frequencies and their amplitudes
        for peak in peaks:
            print(f"Frequency: {xf[peak]} Hz, Amplitude: {yf[peak]}")

        # Save the processed data locally
        processed_df = pd.DataFrame({
            'Time_sec': time_sec,
            'Speed_kph': speed_kph,
            'Filtered_Speed_kph': filtered_speed_kph
        })
        processed_df.to_csv('processed_data.csv', index=False)

        # Upload the processed data back to another S3 bucket
        upload_file_to_s3(bucket, output_key, 'processed_data.csv', s3client)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    run()




if __name__ == "__main__":
    main()
