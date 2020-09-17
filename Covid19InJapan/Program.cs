// Program.cs
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using CsvHelper;
using MathNet.Numerics.Statistics;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using XPlot.Plotly;

namespace Covid19InJapan
{
    internal class Program
    {
        // Filenames of the CSV
        private const string PCR_POSITIVE_DAILY_FILENAME = "pcr_positive_daily.csv";

        private const string PCR_TESTED_DAILY_FILENAME = "pcr_tested_daily.csv";

        // URLs to download the CSV
        private const string PCR_POSITIVE_DAILY_URL = "https://www.mhlw.go.jp/content/pcr_positive_daily.csv";

        private const string PCR_TESTED_DAILY_URL = "https://www.mhlw.go.jp/content/pcr_tested_daily.csv";

        // Window size for moving average
        private const int MA = 7;

        // Main
        private static void Main(string[] args)
        {
            // Download the number of daily PCR positive data and the number of daily PCR tested data from the ministry of health, Japan, if that file does not exist
            using (var wc = new WebClient())
            {
                if (!File.Exists(PCR_POSITIVE_DAILY_FILENAME))
                    wc.DownloadFile(PCR_POSITIVE_DAILY_URL, PCR_POSITIVE_DAILY_FILENAME);
                if (!File.Exists(PCR_TESTED_DAILY_FILENAME))
                    wc.DownloadFile(PCR_TESTED_DAILY_URL, PCR_TESTED_DAILY_FILENAME);
            }

            // Load the both CSV files
            IEnumerable<PCRPositiveDaily> positiveDaily = Enumerable.Empty<PCRPositiveDaily>();
            using (var reader = new StreamReader(PCR_POSITIVE_DAILY_FILENAME, true))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Configuration.RegisterClassMap<PCRPositiveDailyMap>();
                positiveDaily = csv.GetRecords<PCRPositiveDaily>().OrderBy((d) => d.Date).ToArray();
            }
            IEnumerable<PCRTestedDaily> testedDaily = Enumerable.Empty<PCRTestedDaily>();
            using (var reader = new StreamReader(PCR_TESTED_DAILY_FILENAME, true))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Configuration.RegisterClassMap<PCRTestedDailyMap>();
                testedDaily = csv.GetRecords<PCRTestedDaily>().ToArray();
            }

            // Unify the number of daily PCR positive data with the PCR tested data for the time series
            var rawData = positiveDaily.Join(testedDaily, (positive) => positive.Date, (tested) => tested.Date, (positive, tested) =>
            {
                return new { Date = positive.Date, Tested = (double)tested.Count, Positive = (double)positive.Count };
            });

            // Calculate each of the positive rates and convert to an array of the InputData class
            var enmDates = rawData.Select((o) => o.Date).GetEnumerator();
            var enmMATested = rawData.Select((o) => o.Tested).MovingAverage(MA).GetEnumerator();
            var enmMAPositive = rawData.Select((o) => o.Positive).MovingAverage(MA).GetEnumerator();
            IEnumerable<InputData> GetInputData()
            {
                while (enmDates.MoveNext() && enmMATested.MoveNext() && enmMAPositive.MoveNext())
                {
                    var rate = 0f;
                    if (enmMATested.Current > 0)
                        rate = (float)(enmMAPositive.Current / enmMATested.Current) * 100;
                    yield return new InputData { Date = enmDates.Current, PositiveRate = rate };
                }
            }
            var inputData = GetInputData().ToArray();

            // Create the new ML Context
            var mlContext = new MLContext(0);

            // Create a new IDataView from an array of the InputData class
            var data = mlContext.Data.LoadFromEnumerable(inputData);

            // Create a SSA model for forecasting
            var model = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(OutputData.ForecastedPositiveRates),
                inputColumnName: nameof(InputData.PositiveRate),
                windowSize: 14,
                seriesLength: 30,
                trainSize: inputData.Count(),
                horizon: 90,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: nameof(OutputData.LowerBoundPositives),
                confidenceUpperBoundColumn: nameof(OutputData.UpperBoundPositives));

            // To train
            var transformer = model.Fit(data);

            // Create a prediction engine
            var forecastingEngine = transformer.CreateTimeSeriesEngine<InputData, OutputData>(mlContext);

            // Predict the number of positive patients in the next period
            var outputData = forecastingEngine.Predict();

            // Convert the result into chart data
            var actualData = inputData.Select((o) =>
            {
                return new { Date = o.Date, Rate = o.PositiveRate };
            });
            var predictiveData = outputData.ForecastedPositiveRates.Select((o, i) =>
            {
                return new { Date = positiveDaily.Last().Date.AddDays(i + 1), Rate = ReLU(o) };
            });
            var actualGraph = new Graph.Scattergl()
            {
                name = "Actuality",
                x = actualData.Select((o) => o.Date),
                y = actualData.Select((o) => o.Rate),
                mode = "lines+markers"
            };
            var predictiveGraph = new Graph.Scattergl()
            {
                name = "Prediction",
                x = predictiveData.Select((o) => o.Date),
                y = predictiveData.Select((o) => o.Rate),
                mode = "lines+markers"
            };

            // Show the result in a graph
            var chart = Chart.Plot(new[] { actualGraph, predictiveGraph });
            chart.WithTitle("COVID-19 positive rate prediction in Japan");
            chart.WithXTitle("Date");
            chart.WithYTitle("Rate(%)");
            chart.WithSize(800, 800);
            chart.Show();
        }

        // ReLU function
        private static float ReLU(float i) => i < 0 ? 0 : i;
    }
}