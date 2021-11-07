DROP DATABASE predicted_outputs;
CREATE DATABASE predicted_outputs;

DROP TABLE IF EXISTS predicted_outputs;
CREATE TABLE predicted_outputs(
  reason_1 BOOLEAN NOT NULL,
  reason_2 BOOLEAN NOT NULL,
  reason_3 BOOLEAN NOT NULL,
  reason_4 BOOLEAN NOT NULL,
  month_value INT NOT NULL,
  transportation_expense INT NOT NULL,
  age INT NOT NULL,
  body_mass_index INT NOT NULL,
  education BOOLEAN NOT NULL,
  children INT NOT NULL,
  pets INT NOT NULL, 
  probability FLOAT NOT NULL,
  prediction BIT NOT NULL
);