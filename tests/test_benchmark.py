"""Fast network-free tests for the 15-task benchmark registry and leakage rules.

uv run python -m unittest tests.test_benchmark -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"/"rn_interlude"))
sys.path.insert(0, str(ROOT/"experiments"/"datasets_ex03"))

from benchmark_datasets import TASKS, load_task, profile_train
from benchmark import metrics
from datasets import preprocess


class TestBenchmark(unittest.TestCase):
    def test_registry_has_exactly_fifteen_unique_tasks(self):
        self.assertEqual(len(TASKS), 15)
        self.assertEqual(len(set(TASKS)), 15)
        for name, spec in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(spec.slug)
                self.assertTrue(spec.filename)
                self.assertTrue(spec.target)
                self.assertTrue(spec.domain)

    def test_metrics_in_train_standardized_units(self):
        y=np.array([1.,2.,3.])
        p=np.array([1.,2.,4.])
        m=metrics(y,p,2.)
        self.assertAlmostEqual(m["RMSE"],1/np.sqrt(3))
        self.assertAlmostEqual(m["NRMSE"],1/(2*np.sqrt(3)))
        self.assertAlmostEqual(m["R2"],.5)

    def write_task(self,tmp,name,rows,filename):
        path=Path(tmp)/name
        path.mkdir(parents=True)
        frame=pd.DataFrame(rows)
        frame.to_csv(path/filename,index=False)
        return path

    def test_insurance_data_train_only_preprocessing(self):
        with tempfile.TemporaryDirectory() as root:
            n=180
            k=np.arange(n)
            frame=dict(age=20+(k%50),bmi=20+.1*k,children=k%5,
                sex=np.where(k%2,"male","female"),
                smoker=np.where(k%7==0,"yes","no"),
                region=np.where(k%3==0,"northwest","southeast"),
                charges=2500+23*k+5000*(k%7==0))
            self.write_task(root,"insurance",frame,"insurance.csv")
            b=load_task("insurance",data_root=Path(root))
            self.assertEqual(len(b["numeric"]),3)
            self.assertEqual(len(b["categorical"]),3)
            self.assertNotIn("charges",b["features"])
            d=preprocess(b,42)
            self.assertEqual(sum(len(x["indices"]) for x in d["splits"].values()),n)
            np.testing.assert_allclose(d["splits"]["train"]["numeric"].mean(axis=0),
                np.zeros(3),atol=1e-5)
            self.assertEqual(d["splits"]["train"]["tensor"].tensors[0].shape[1],3)
            p=profile_train(d,b,42)
            self.assertEqual(p["rows"],n)
            self.assertEqual(p["features"],6)

    def test_target_components_are_excluded_for_bikes(self):
        with tempfile.TemporaryDirectory() as root:
            n=180
            k=np.arange(n)
            frame=dict(instant=k+1,dteday=["2012-01-01"]*n,
                season=k%4+1,yr=k%2,mnth=k%12+1,hr=k%24,
                holiday=k%2,weekday=k%7,workingday=k%2,weathersit=k%4+1,
                temp=(k%24)/24,atemp=(k%20)/20,hum=(k%10)/10,
                windspeed=(k%5)/5,casual=k%20+2,registered=k%30+10)
            frame["cnt"]=frame["casual"]+frame["registered"]
            self.write_task(root,"bike_hourly",frame,"hour.csv")
            b=load_task("bike_hourly",data_root=Path(root))
            for forbidden in ("instant","dteday","casual","registered","cnt"):
                self.assertNotIn(forbidden,b["features"])
            self.assertIn("weathersit",b["categorical"])

    def test_student_prior_grades_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            n=180
            k=np.arange(n)
            frame=dict(school=np.where(k%2,"GP","MS"),sex=np.where(k%2,"F","M"),
                age=k%5+15,studytime=k%4+1,G1=k%20,G2=k%20,G3=k%20)
            path=Path(root)/"student_math"
            path.mkdir(parents=True)
            pd.DataFrame(frame).to_csv(path/"student-mat.csv",sep=";",index=False)
            b=load_task("student_math",data_root=Path(root))
            for forbidden in ("G1","G2","G3"):
                self.assertNotIn(forbidden,b["features"])
            self.assertIn("sex",b["categorical"])

    def test_auto_mpg_question_mark_is_numeric_missingness(self):
        with tempfile.TemporaryDirectory() as root:
            n=180
            k=np.arange(n)
            hp=np.where(k==0,"?","125")
            frame=dict(mpg=20+k%20,cylinders=k%3+4,displacement=100+k,
                horsepower=hp,weight=1500+10*k,acceleration=11+k%5,
                model_year=70+k%12,origin=k%3+1)
            frame["car name"]=["synthetic model"]*n
            self.write_task(root,"auto_mpg",frame,"auto-mpg.csv")
            b=load_task("auto_mpg",data_root=Path(root))
            self.assertIn("horsepower",b["numeric"])
            self.assertTrue(np.isnan(b["frame"]["horsepower"].iloc[0]))
            self.assertNotIn("car name",b["features"])
            self.assertIn("origin",b["categorical"])

    def test_california_local_path_retains_existing_ex03_schema(self):
        with tempfile.TemporaryDirectory() as root:
            n=180
            k=np.arange(n)
            frame=dict(longitude=-123+k%10/10,latitude=33+k%20/10,
                housing_median_age=10+k%40,total_rooms=500+k*5,
                total_bedrooms=200+k,population=2000+k*3,
                households=300+k,median_income=1.+k%10,
                median_house_value=80000.+k*1000.,
                ocean_proximity=np.where(k%2,"INLAND","NEAR OCEAN"))
            self.write_task(root,"california",frame,"housing.csv")
            b=load_task("california",data_root=Path(root))
            self.assertEqual(len(b["features"]),9)
            self.assertEqual(len(b["numeric"]),8)
            self.assertEqual(len(b["categorical"]),1)
            self.assertNotIn("median_house_value",b["features"])


if __name__=="__main__":
    unittest.main()
