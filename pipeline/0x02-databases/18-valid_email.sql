-- This script creates a trigger that resets the attribute valid_email only when the email has been changed
CREATE TRIGGER `user_uu` BEFORE UPDATE ON `users` FOR EACH ROW IF OLD.email <> NEW.email THEN SET NEW.valid_email = 0; END IF;
