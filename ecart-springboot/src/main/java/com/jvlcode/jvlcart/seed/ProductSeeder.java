package com.jvlcode.jvlcart.seed;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.stereotype.Component;

import com.jvlcode.jvlcart.entity.Product;
import com.jvlcode.jvlcart.repository.ProductRepository;

@Component
public class ProductSeeder implements CommandLineRunner {
	
	@Autowired
	private ProductRepository productRepository;

	@Override
	public void run(String... args) throws Exception {
		if (productRepository.count() == 0) {
			List<Product> demoProducts = List.of(
				    new Product(null, "Apple iPhone 15", 799.0, "SmartPhone with A16 Chip", "Phones", 4.8, "Amazon", 5, List.of("/products/1.jpg")),
				    new Product(null, "Samsung Galaxy S23", 749.0, "Flagship Android with Snapdragon 8 Gen 2", "Phones", 4.7, "Flipkart", 8, List.of("/products/2.jpg")),
				    new Product(null, "Dell XPS 13", 1199.0, "Ultrabook with Intel i7 and 16GB RAM", "Laptops", 4.6, "Amazon", 3, List.of("/products/3.jpg")),
				    new Product(null, "Sony WH-1000XM5", 349.0, "Noise Cancelling Wireless Headphones", "Audio", 4.9, "Croma", 10, List.of("/products/4.jpg")),
				    new Product(null, "Apple MacBook Air M2", 999.0, "Lightweight laptop with M2 chip", "Laptops", 4.8, "Amazon", 4, List.of("/products/5.jpg")),
				    new Product(null, "Canon EOS R50", 679.0, "Mirrorless Camera with 24MP sensor", "Cameras", 4.5, "Reliance Digital", 2, List.of("/products/6.jpg")),
				    new Product(null, "OnePlus 11", 699.0, "Flagship killer with Snapdragon 8 Gen 2", "Phones", 4.6, "Amazon", 6, List.of("/products/7.jpg")),
				    new Product(null, "JBL Flip 6", 129.0, "Portable Bluetooth Speaker", "Audio", 4.4, "Flipkart", 12, List.of("/products/8.jpg")),
				    new Product(null, "HP Envy x360", 899.0, "2-in-1 Laptop with Ryzen 7", "Laptops", 4.7, "Amazon", 5, List.of("/products/9.jpg")),
				    new Product(null, "Google Pixel 7", 649.0, "Clean Android experience with Tensor chip", "Phones", 4.5, "Flipkart", 7, List.of("/products/10.jpg"))
				);
			productRepository.saveAll(demoProducts);
		     System.out.println("✅ Seeded demo products.");
        } else {
            System.out.println("✔️ Products already exist. Skipping seed.");
        }
		
	}

}
