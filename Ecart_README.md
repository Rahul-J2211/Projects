# 🛒 E-Cart Spring Boot Backend

A scalable E-Commerce REST API built using Java and Spring Boot.\
This project demonstrates backend development skills including layered
architecture, database integration, and RESTful API design.

------------------------------------------------------------------------

## 📌 Project Overview

E-Cart is a backend application designed to handle core e-commerce
functionalities such as:

-   Product Management
-   Cart Handling
-   Order Processing
-   Database Persistence
-   Clean MVC Architecture

This project focuses on backend logic and API design rather than
frontend UI.

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   Java
-   Spring Boot
-   Spring Web (REST APIs)
-   Spring Data JPA
-   MySQL
-   Maven

------------------------------------------------------------------------

## 🧱 Architecture

The project follows a layered architecture:

Controller → Service → Repository → Database

### Structure

    ecart-springboot
    │
    ├── controller        # REST endpoints
    ├── service           # Business logic
    ├── repository        # JPA repositories
    ├── model             # Entity classes
    ├── resources
    │   └── application.properties
    └── pom.xml

------------------------------------------------------------------------

## 🚀 Features

### 🛍️ Product APIs

-   Create Product
-   Get All Products
-   Get Product by ID
-   Update Product
-   Delete Product

### 🛒 Cart / Order Logic

-   Add items to cart
-   Update cart
-   Order placement logic

### 🗄️ Database Integration

-   MySQL configuration
-   Hibernate ORM mapping
-   Auto table generation (JPA)

------------------------------------------------------------------------

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

    git clone https://github.com/Rahul-J2211/Projects.git
    cd Projects/ecart-springboot

### 2️⃣ Configure Database

Create database:

    CREATE DATABASE ecart_db;

Update `application.properties`:

    spring.datasource.url=jdbc:mysql://localhost:3306/ecart_db
    spring.datasource.username=YOUR_USERNAME
    spring.datasource.password=YOUR_PASSWORD

    spring.jpa.hibernate.ddl-auto=update
    spring.jpa.show-sql=true

### 3️⃣ Run Application

    mvn spring-boot:run

Server runs at:

    http://localhost:8080

------------------------------------------------------------------------

## 🧪 API Testing

You can test APIs using:

-   Postman
-   Swagger (if enabled)
-   cURL

------------------------------------------------------------------------

## 📈 Future Enhancements

-   JWT Authentication & Role-Based Access
-   Pagination & Sorting
-   Payment Gateway Integration
-   Swagger API Documentation
-   Unit & Integration Testing
-   Deployment to Cloud (AWS / Render)

------------------------------------------------------------------------

## 👨‍💻 Author

Rahul J\
Java Backend Developer \| Spring Boot \| REST APIs \| MySQL
