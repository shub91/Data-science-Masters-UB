# Project 2

CREATE database Proj2;
USE Proj2;

# Question 1

# Creating Tables

CREATE TABLE Employee (Ename VARCHAR (30) NOT NULL, Salary INT NOT NULL, PRIMARY KEY (Ename));

CREATE TABLE Project (Pname VARCHAR (30) NOT NULL, Agency VARCHAR (30) NOT NULL, Budget INT NOT NULL, PRIMARY KEY (Pname));

CREATE TABLE Assign(Ename VARCHAR (30) NOT NULL, Pname VARCHAR (30) NOT NULL, PRIMARY KEY (Ename, Pname), FOREIGN KEY (Ename) REFERENCES Employee (Ename),
FOREIGN KEY (Pname) REFERENCES Project (Pname));

# Inserting Sample Data to test queries

INSERT INTO Project (Pname, Agency, Budget)
VALUES ('P1', 'A1', 100), ('P2', 'A2', 200), ('P3', 'A1', 300), ('P4', 'A1', 400);
    

INSERT INTO Employee (Ename, Salary)
VALUES ('Mark', 10), ('Mary', 20), ('Mike', 30), ('Aby', 40);

    
INSERT INTO Assign (Ename, Pname)
VALUES  ('Mark', 'P1'), ('Mary', 'P1'), ('Mike', 'P4'), ('Aby', 'P1'), ('Aby', 'P2');

# Queries for given question
# S1

SELECT Ename
FROM Assign
GROUP BY Ename HAVING COUNT(DISTINCT Pname) = 1;

# S2

SELECT Ename
FROM Employee
WHERE Salary > (SELECT Salary FROM Employee WHERE Ename = 'Mark');

# S3

SELECT P1.Pname, count(P2.Pname) AS PROJ_NUM_W_HIGHER_BUDGET
FROM Project P1 LEFT JOIN Project P2 ON P2.Budget > P1.Budget
GROUP BY P1.Pname;

# S4

SELECT F.Pname2 FROM (SELECT P1.Pname AS Pname2, P1.Agency AS Agency2, P1.Budget AS Budget2, A.Budget_avg AS Budget_avg2
FROM Project P1 LEFT JOIN (SELECT Agency, AVG(Budget) AS Budget_avg FROM Project GROUP BY Agency) A
ON P1.Agency = A.Agency) F WHERE F.Budget2 < F.Budget_avg2;

# Question 2

# Creating Tables/ Schema

CREATE TABLE IF NOT EXISTS Nodetable (Nodename CHAR(1) NOT NULL, PRIMARY KEY (Nodename)) ;

CREATE TABLE IF NOT EXISTS Network_DAG (Node_init CHAR(1) NOT NULL, Node_Fin CHAR(1) NOT NULL,
FOREIGN KEY (Node_init) REFERENCES Nodetable(Nodename),
FOREIGN KEY (Node_Fin) REFERENCES Nodetable(Nodename));

# Creating a sample graph

INSERT INTO Nodetable(Nodename)
VALUES ('A'), ('B'), ('C'), ('D'), ('E');
        
INSERT INTO Network_DAG (Node_init, Node_fin)
VALUES  ('A','B'), ('B','C'), ('C','A'), ('A','C'), ('D','E');

# Queries for given question

# Part 1

SELECT T.Nodename, IFNULL(I.COUNTS,0) AS NUM_OF_NODES_DIRECTLY_REACHABLE
FROM Nodetable T LEFT JOIN 
(SELECT Node_init, COUNT(*) COUNTS
FROM Network_DAG
GROUP BY Node_init) I
ON T.Nodename = I.Node_init;

# Part 2

WITH RECURSIVE path_d(Node_init, Node_fin) AS( 
(SELECT Node_init, Node_fin FROM Network_DAG)
UNION DISTINCT
(SELECT F.Node_init, D.Node_fin
FROM Network_DAG D, path_d F
WHERE F.Node_fin = D.Node_init))

SELECT A.Nodename, IFNULL(J.COUNTS,0) AS DIRECTLY_INDIRECTLY_REACHABLE
FROM Nodetable A LEFT JOIN 
(SELECT Node_init, COUNT(*) COUNTS
FROM path_d F
GROUP BY Node_init) J
ON A.Nodename = J.Node_init;

# Part 3

SELECT *
from Nodetable Q
WHERE NOT EXISTS(
SELECT 'A'
FROM Network_DAG
WHERE Node_init = 'A' AND Node_fin = Q.Nodename );

# Part 4

WITH RECURSIVE path_d(Node_init, Node_fin) AS(
(SELECT Node_init, Node_fin FROM Network_DAG WHERE Node_init ='A')
UNION
(SELECT D.Node_fin, F.Node_fin
FROM Network_DAG F, path_d D
WHERE D.Node_fin = F.Node_init))

SELECT * FROM Nodetable WHERE NOT EXISTS(
SELECT Node_fin FROM path_d WHERE path_d.Node_fin = Nodetable.Nodename) and Nodename <>'A';

select * from Network_DAG;

# Question 3

CREATE TABLE R(A1, ... , AN);

CREATE VIEW COL(A1, ... , AN, IND, V) AS
	(SELECT *, 1, A1
	FROM R)
	UNION
	(SELECT *, 2, A2
	FROM R)
	UNION
	...
	UNION
	(SELECT *, N, AN
	FROM R);

CREATE VIEW VAL(A1, ... ,AN,Ct) AS
	SELECT A1, ... , AN, COUNT(DISTINCT V) FROM COL
	GROUP BY A1, ... , AN;

SELECT A1, ... , AN FROM VAL
WHERE Ct = SELECT MIN(Ct) FROM VAL;