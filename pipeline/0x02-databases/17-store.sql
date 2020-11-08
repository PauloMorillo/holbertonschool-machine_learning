-- This script creates a trigger that decreases the quantity of an item after adding a new order
CREATE TRIGGER `orders_ui` AFTER INSERT ON `orders` FOR EACH ROW UPDATE items SET items.quantity=items.quantity - NEW.number WHERE items.name = NEW.item_name AND items.quantity >= NEW.number
