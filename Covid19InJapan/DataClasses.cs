// DataClasses.cs
using System;
using CsvHelper.Configuration;

namespace Covid19InJapan
{
    // Data class for pcr_tested_daily.csv
    public class PCRTestedDaily
    {
        public DateTime Date { get; set; }

        public int Count { get; set; }
    }

    // Data class for pcr_positive_daily.csv
    public class PCRPositiveDaily
    {
        public DateTime Date { get; set; }

        public int Count { get; set; }
    }

    // Map pcr_tested_daily.csv to PCRTestedDaily class
    public class PCRTestedDailyMap : ClassMap<PCRTestedDaily>
    {
        public PCRTestedDailyMap()
        {
            Map(m => m.Date).Index(0);
            Map(m => m.Count).Index(1);
        }
    }

    // Map pcr_positive_daily.csv to PCRPositiveDaily class
    public class PCRPositiveDailyMap : ClassMap<PCRPositiveDaily>
    {
        public PCRPositiveDailyMap()
        {
            Map(m => m.Date).Index(0);
            Map(m => m.Count).Index(1);
        }
    }

    // Input data for training
    public class InputData
    {
        // Date
        public DateTime Date { get; set; }

        // The number of positive patients
        public float PositiveRate { get; set; }

        // For debugging
        public override string ToString()
        {
            return $"{PositiveRate}";
        }
    }

    // Results of prediction engine
    public class OutputData
    {
        // The number of positive persons in the prediction period.
        public float[] ForecastedPositiveRates { get; set; }

        // The minumum number of positive persons in the prediction period.
        public float[] LowerBoundPositives { get; set; }

        // The maximum number of positive persons in the prediction period.
        public float[] UpperBoundPositives { get; set; }
    }
}