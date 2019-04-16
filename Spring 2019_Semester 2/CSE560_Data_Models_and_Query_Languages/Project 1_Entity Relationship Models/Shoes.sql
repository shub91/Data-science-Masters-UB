create database ShoesDB;
use ShoesDB;

create table Bus_Cust(
	username varchar(20),
    passwd varchar(25) not null,
    first_name varchar(20) not null,
    last_name varchar(20) not null,
    street varchar(40) not null,
    city varchar(20) not null,
    state varchar(20) not null,
    zip int not null check (zip between 0 and 99999),
    country varchar(20) not null,
    company_name varchar(40) not null,
    primary key (username)
);

create table Ind_Users(
	username varchar(20),
    passwd varchar(25) not null,
    first_name varchar(20) not null,
    last_name varchar(20) not null,
    street varchar(40) not null,
    city varchar(20) not null,
    state varchar(20) not null,
    zip int not null check (zip between 0 and 99999),
    country varchar(20) not null,
    primary key (username)
);

create table Orders(
	order_num varchar(20),
    total_amount float not null check (total_amount >= 0),
    o_discount float not null check (o_discount between 0 and 100),
    username varchar(20) not null,
    primary key (order_num),
    foreign key (username) references Ind_Users(username),
    foreign key (username) references Bus_Cust(username)
);

create table Product(
	prod_id varchar(20),
    p_discount float not null check (p_discount between 0 and 100),
    price float not null check (price >= 0),
    shoe_name varchar(20) not null,
    category varchar(20) not null,
    descript varchar(100) not null,
    primary key (prod_id)
);

create table OrderProduct(
	o_quantity int default 0 check (o_quantity >= 0),
    r_quantity int default 0 check (r_quantity >= 0 and r_quantity <= o_quantity),
    p_status int default 0 check (p_status in (0, 1, 2, 3, 4)),
    prod_id varchar(20) not null,
    order_num varchar(20) not null,
    foreign key (prod_id) references Product(prod_id),
    foreign key (order_num) references Orders(order_num)
);


create table Availability(
	prod_id varchar(20),
    size int check (size between 1 and 20),
    quantity int check (quantity > 0),
    primary key (size),
    foreign key (prod_id) references Product(prod_id)
);