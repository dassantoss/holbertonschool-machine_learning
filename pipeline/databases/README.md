# Databases

## Overview

This project focuses on working with both relational (SQL/MySQL) and non-relational (NoSQL/MongoDB) databases for storing and manipulating data. It covers fundamental operations such as creating databases, tables, and indexes, as well as advanced features like views, triggers, and stored procedures in MySQL. It also explores performing CRUD operations and queries in MongoDB, and interacting with MongoDB from Python using PyMongo.

## Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General
- What's a relational database
- What's a none relational database
- What is difference between SQL and NoSQL
- How to create tables with constraints
- How to optimize queries by adding indexes
- What is and how to implement stored procedures and functions in MySQL
- What is and how to implement views in MySQL
- What is and how to implement triggers in MySQL
- What is ACID
- What is a document storage
- What are NoSQL types
- What are benefits of a NoSQL database
- How to query information from a NoSQL database
- How to insert/update/delete information from a NoSQL database
- How to use MongoDB

## Resources
### Read or watch:

#### MySQL
- [What is Database & SQL?](https://www.youtube.com/watch?v=FR4QIeZaPeM)
- [MySQL Cheat Sheet](https://devhints.io/mysql)
- [MySQL 5.7 SQL Statement Syntax](https://dev.mysql.com/doc/refman/5.7/en/sql-statements.html)
- [MySQL Performance: How To Leverage MySQL Database Indexing](https://www.liquidweb.com/kb/mysql-optimization-how-to-leverage-mysql-database-indexing/)
- [Stored Procedure](https://www.w3resource.com/mysql/mysql-procedure.php)
- [Triggers](https://www.w3resource.com/mysql/mysql-triggers.php)
- [Views](https://www.w3resource.com/mysql/mysql-views.php)
- [Functions and Operators](https://dev.mysql.com/doc/refman/5.7/en/functions.html)
- [Trigger Syntax and Examples](https://dev.mysql.com/doc/refman/5.7/en/trigger-syntax.html)
- [CREATE TABLE Statement](https://dev.mysql.com/doc/refman/5.7/en/create-table.html)
- [CREATE PROCEDURE and CREATE FUNCTION Statements](https://dev.mysql.com/doc/refman/5.7/en/create-procedure.html)
- [CREATE INDEX Statement](https://dev.mysql.com/doc/refman/5.7/en/create-index.html)
- [CREATE VIEW Statement](https://dev.mysql.com/doc/refman/5.7/en/create-view.html)

#### NoSQL
- [NoSQL Databases Explained](https://riak.com/resources/nosql-databases/)
- [What is NoSQL?](https://www.youtube.com/watch?v=qUV2j3XBRHc)
- [Building Your First Application: An Introduction to MongoDB](https://www.mongodb.com/blog/post/building-your-first-application-mongodb-creating-rest-api-using-mean-stack-part-1)
- [MongoDB Tutorial 2: Insert, Update, Remove, Query](https://www.youtube.com/watch?v=CB9G5Dvv-EE)
- [Aggregation](https://docs.mongodb.com/manual/aggregation/)
- [Introduction to MongoDB and Python](https://realpython.com/introduction-to-mongodb-and-python/)
- [mongo Shell Methods](https://docs.mongodb.com/manual/reference/method/)
- [The mongo Shell](https://docs.mongodb.com/manual/mongo/)

## Requirements

### General
- A README.md file, at the root of the folder of the project, is mandatory
- All your SQL files will be executed on Ubuntu 20.04 LTS using MySQL 8.0 (version 8.0.39)
- All your SQL queries should have a comment just before
- All SQL keywords should be in uppercase (SELECT, WHERE...)
- All your Mongo files will be interpreted/compiled on Ubuntu 20.04 LTS using MongoDB (version 4.4.29)
- The first line of all your Mongo files should be a comment: `// my comment`
- All your Python files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.9) and PyMongo (version 4.6.2)
- The first line of all your Python files should be exactly `#!/usr/bin/env python3`
- Your Python code should use the pycodestyle style (version 2.11.1)
- All your Python modules should have documentation
- All your Python functions should have documentation
- Your Python code should not be executed when imported

### Installation

#### MySQL 8.0
```bash
sudo apt-get update
sudo apt-get install mysql-server
service mysql start
mysql -uroot -p
```

#### MongoDB 4.4
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-4.4.list
sudo apt-get update
sudo apt-get install -y mongodb-org
service mongod start
```

#### Python Dependencies
```bash
sudo apt-get install python3-pip
pip3 install pymongo==4.6.2
pip3 install pycodestyle==2.11.1
```

## Project Structure

### SQL Tasks (MySQL)
- Database creation and management
- Table operations and constraints
- Query optimization with indexes
- Stored procedures and functions
- Views and triggers
- Advanced SQL operations

### NoSQL Tasks (MongoDB)
- Basic CRUD operations
- Document queries and manipulation
- Python integration with PyMongo
- Log analysis and statistics

## Code Style and Conventions

### SQL Files
- Must have comments before queries
- Keywords in uppercase
- Will be executed on MySQL 8.0

### MongoDB Files
- Must start with a comment
- Will be interpreted using MongoDB 4.4

### Python Files
- Must start with shebang line
- Must include documentation
- Must follow pycodestyle
- Should not execute when imported

## Additional Resources

### MySQL
- [MySQL Documentation](https://dev.mysql.com/doc/)
- [MySQL Performance: Database Indexing](https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html)
- [Stored Procedures](https://dev.mysql.com/doc/refman/8.0/en/stored-programs-defining.html)

### MongoDB
- [MongoDB Manual](https://docs.mongodb.com/manual/)
- [PyMongo Documentation](https://pymongo.readthedocs.io/)
- [MongoDB Shell Methods](https://docs.mongodb.com/manual/reference/method/)

## Tasks

### 0. Create a database
Write a script that creates the database db_0 in your MySQL server.
* If the database db_0 already exists, your script should not fail
* You are not allowed to use the SELECT or SHOW statements
* File: `0-create_database_if_missing.sql`

### 1. First table
Write a script that creates a table called first_table in the current database.
* first_table description:
  * id INT
  * name VARCHAR(256)
* The database name will be passed as an argument
* If the table exists, script should not fail
* File: `1-first_table.sql`

### 2. List all in table
Write a script that lists all rows of the table first_table.
* All fields should be printed
* The database name will be passed as an argument
* File: `2-list_values.sql`

### 3. First add
Write a script that inserts a new row in the table first_table.
* New row:
  * id = 89
  * name = Holberton School
* The database name will be passed as an argument
* File: `3-insert_value.sql`

### 4. Select the best
Write a script that lists all records with score >= 10 in second_table.
* Results should display score and name (in this order)
* Records should be ordered by score (top first)
* File: `4-best_score.sql`

### 5. Average
Write a script that computes the score average of all records in second_table.
* The result column name should be average
* File: `5-average.sql`

### 6. Temperatures #0
Write a script that displays the average temperature by city ordered by temperature.
* Import table dump provided
* File: `6-avg_temperatures.sql`

### 7. Temperatures #2  
Write a script that displays the max temperature of each state.
* Results ordered by State name
* File: `7-max_state.sql`

### 8. Genre ID by show
Write a script that lists all shows with at least one genre linked.
* Display: tv_shows.title - tv_show_genres.genre_id
* Results sorted by tv_shows.title and tv_show_genres.genre_id
* File: `8-genre_id_by_show.sql`

### 9. No genre
Write a script that lists all shows without a genre linked.
* Display: tv_shows.title - tv_show_genres.genre_id
* Results sorted by tv_shows.title and tv_show_genres.genre_id
* File: `9-no_genre.sql`

### 10. Number of shows by genre
Write a script that lists all genres and their number of linked shows.
* Display: genre - number_of_shows
* Don't display genres without linked shows
* Results sorted by descending number of shows
* File: `10-count_shows_by_genre.sql`

### 11. Rotten tomatoes
Write a script that lists all shows by their rating.
* Display: tv_shows.title - rating sum
* Results sorted in descending order by rating
* File: `11-rating_shows.sql`

### 12. Best genre
Write a script that lists all genres by their rating.
* Display: tv_genres.name - rating sum
* Results sorted in descending order by rating
* File: `12-rating_genres.sql`

### 13. We are all unique!
Write a SQL script that creates a table users with:
* id: integer, primary key, auto increment
* email: string (255), unique, not null
* name: string (255)
* File: `13-uniq_users.sql`

### 14. In and not out
Write a SQL script that creates a table users with:
* Additional country field: enumeration of US, CO and TN
* Default country value: US
* File: `14-country_users.sql`

### 15. Best band ever!
Write a SQL script that ranks country origins of bands by number of fans.
* Columns: origin and nb_fans
* Results sorted by nb_fans
* File: `15-fans.sql`

### 16. Old school band
Write a SQL script that lists Glam rock bands by longevity.
* Display: band_name and lifespan (until 2020)
* Use formed and split fields
* File: `16-glam_rock.sql`

### 17. Buy buy buy
Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.
* Quantity in the table items can be negative
* File: `17-store.sql`

### 18. Email validation to sent
Write a SQL script that creates a trigger that resets the attribute valid_email only when the email has been changed.
* File: `18-valid_email.sql`

### 19. Add bonus
Write a SQL script that creates a stored procedure AddBonus.
* Procedure takes 3 inputs:
  * user_id: users.id value
  * project_name: new or existing project
  * score: score value for correction
* File: `19-bonus.sql`

### 20. Average score
Write a SQL script that creates a stored procedure ComputeAverageScoreForUser.
* Procedure computes and stores the average score for a student
* Takes 1 input: user_id
* File: `20-average_score.sql`

### 21. Safe divide
Write a SQL script that creates a function SafeDiv that divides two numbers.
* Function takes 2 arguments: a and b (INT)
* Returns a/b or 0 if b == 0
* File: `21-div.sql`

### 22. List all databases
Write a script that lists all databases in MongoDB.
* File: `22-list_databases`

### 23. Create a database
Write a script that creates or uses the database my_db.
* File: `23-use_or_create_database`

### 24. Insert document
Write a script that inserts a document in the collection school.
* Document must have attribute name="Holberton school"
* Database name passed as mongo command option
* File: `24-insert`

### 25. All documents
Write a script that lists all documents in the collection school.
* Database name passed as mongo command option
* File: `25-all`

### 26. All matches
Write a script that lists all documents with name="Holberton school".
* Database name passed as mongo command option
* File: `26-match`

### 27. Count
Write a script that displays the number of documents in collection school.
* Database name passed as mongo command option
* File: `27-count`

## MongoDB Tasks

### 28. Update document
Write a script that adds a new attribute to a document in collection school:
* Update only documents with name="Holberton school"
* Add attribute address with value "972 Mission street"
* File: `28-update`

### 29. Delete by match
Write a script that deletes all documents with name="Holberton school" in collection school.
* File: `29-delete`

### 30. List all documents in Python
Write a Python function that lists all documents in a collection:
* Prototype: `def list_all(mongo_collection):`
* Return empty list if no documents
* File: `30-all.py`

### 31. Insert a document in Python
Write a Python function that inserts a new document in a collection:
* Prototype: `def insert_school(mongo_collection, **kwargs):`
* Returns the new _id
* File: `31-insert_school.py`

### 32. Change school topics
Write a Python function that changes all topics of a school document:
* Prototype: `def update_topics(mongo_collection, name, topics):`
* name (string) will be the school name to update
* topics (list of strings) will be the list of topics approached in the school
* File: `32-update_topics.py`

### 33. Where can I learn Python?
Write a Python function that returns the list of schools having a specific topic:
* Prototype: `def schools_by_topic(mongo_collection, topic):`
* topic (string) will be topic searched
* File: `33-schools_by_topic.py`

### 34. Log stats
Write a Python script that provides stats about Nginx logs stored in MongoDB:
* Database: logs
* Collection: nginx
* Display:
  * First line: x logs where x is number of documents
  * Second line: Methods with count of documents for each
  * One line with number of documents with method=GET and path=/status
* File: `34-log_stats.py`

## Advanced Tasks

### 35. Optimize simple search
Write a SQL script that creates an index idx_name_first on the table names.
* Only the first letter of name must be indexed
* Import table dump provided: names.sql.zip
* File: `100-index_my_names.sql`

### 36. Optimize search and score
Write a SQL script that creates an index idx_name_first_score.
* Index should be on first letter of name AND score
* Import table dump provided: names.sql.zip
* File: `101-index_name_score.sql`

### 37. No table for a meeting
Write a SQL script that creates a view need_meeting listing students that:
* Have a score under 80 (strict)
* AND no last_meeting date OR more than 1 month
* File: `102-need_meeting.sql`

### 38. Average weighted score
Write a SQL script that creates a stored procedure ComputeAverageWeightedScoreForUser.
* Procedure computes and stores the average weighted score for a student
* Takes user_id as input
* File: `103-average_weighted_score.sql`

### 39. Regex filter
Write a script that lists all documents with name starting by "Holberton" in collection school.
* Database name passed as option of mongo command
* File: `104-find`

### 40. Top students
Write a Python function that returns all students sorted by average score.
* Prototype: `def top_students(mongo_collection):`
* mongo_collection will be the pymongo collection object
* Must be ordered by average score
* Each item must include averageScore key
* File: `105-students.py`

### 41. Log stats - new version
Improve 34-log_stats.py by adding the top 10 most present IPs in nginx collection.
* IPs must be sorted
* File: `106-log_stats.py`

## Author
- Alexa Orrico, Software Engineer at Holberton School

## License
This project is part of the curriculum of Holberton School. All rights reserved.
