-- This script creates  table called users in the current database
CREATE TABLE IF NOT EXISTS users (
id integer NOT NULL AUTO_INCREMENT,
email VARCHAR(255) NOT NULL UNIQUE,
name VARCHAR(255),
PRIMARY KEY (id)
);
