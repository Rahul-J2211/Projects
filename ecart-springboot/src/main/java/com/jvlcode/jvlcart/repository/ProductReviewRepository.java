package com.jvlcode.jvlcart.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.jvlcode.jvlcart.entity.Product;
import com.jvlcode.jvlcart.entity.ProductReview;

public interface ProductReviewRepository  extends JpaRepository<ProductReview, Long>{
	
}
