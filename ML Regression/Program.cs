using System;
using System.ComponentModel;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ML_Regression
{
    class Program
    {
        private static string _path = @"..\..\..\Data\boston.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(0);

            var textLoaderOptions = new TextLoader.Options()
            {
                Separators = new[] { ',' },
                HasHeader = true
            };

            var loader = context.Data.CreateTextLoader<Input>(textLoaderOptions);
            var data = loader.Load(_path);

            var splitedData = context.Data.TrainTestSplit(data, 0.2D);
            var testData = splitedData.TestSet;
            var trainData = splitedData.TrainSet;

            var pipeline = context.Transforms.Concatenate("Features", "CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT")
                .Append(context.Regression.Trainers.FastTreeTweedie());

            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = context.Regression.Evaluate(predictions);
            Console.WriteLine($"Evaluation of 20% of data: {metrics.RSquared:0.00#}");

            var scores = context.Regression.CrossValidate(data, pipeline, 5);
            var average = scores.Average(x => x.Metrics.RSquared);
            Console.WriteLine($"Cross validation data with 5 iterations: {average:0.00#}");


            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            var dummpyInput = new Input()
            {
                CRIM = 0.00632F,
                ZN = 12.00F,
                INDUS = 1.310F,
                CHAS = 0,
                NOX = 0.8380F,
                RM = 5.5750F,
                AGE = 68.20F,
                DIS = 5.0900F,
                RAD = 1,
                TAX = 295.0F,
                PTRATIO = 14.30F,
                B = 398.90F,
                LSTAT = 5.0F
            };

            var prediction = predictor.Predict(dummpyInput);
            Console.WriteLine($"Prediction value of owner-occupied home is: {prediction.Price}");
            Console.ReadKey();
        }


        private class Input
        {
            [Description("per capita crime rate by town")]
            [LoadColumn(0)]
            public float CRIM;

            [Description("proportion of residential land zoned for lots over 25,000 sq.ft")]
            [LoadColumn(1)]
            public float ZN;

            [Description("proportion of non-retail business acres per town")]
            [LoadColumn(2)]
            public float INDUS;

            [Description("Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)")]
            [LoadColumn(3)]
            public float CHAS;

            [Description("nitric oxides concentration (parts per 10 million)")]
            [LoadColumn(4)]
            public float NOX;

            [Description("average number of rooms per dwelling")]
            [LoadColumn(5)]
            public float RM;

            [Description("proportion of owner-occupied units built prior to 1940")]
            [LoadColumn(6)]
            public float AGE;

            [Description("weighted distances to five Boston employment centres")]
            [LoadColumn(7)]
            public float DIS;

            [Description("index of accessibility to radial highways")]
            [LoadColumn(8)]
            public float RAD;

            [Description("full-value property-tax rate per $10,000")]
            [LoadColumn(9)]
            public float TAX;

            [Description("pupil-teacher ratio by town")]
            [LoadColumn(10)]
            public float PTRATIO;

            [Description("1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")]
            [LoadColumn(11)]
            public float B;

            [Description("% lower status of the population")]
            [LoadColumn(12)]
            public float LSTAT;

            [Description("Median value of owner-occupied homes in $1000's")]
            [LoadColumn(13), ColumnName("Label")]
            public float MEDV;
        }

        private class Output
        {
            [ColumnName("Score")]
            public float Price;
        }
    }
}
