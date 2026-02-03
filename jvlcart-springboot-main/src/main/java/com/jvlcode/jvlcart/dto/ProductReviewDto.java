package com.jvlcode.jvlcart.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

public class ProductReviewDto {
	@NotNull(message = "Product ID is required")
	private Long productId;
	public Long getProductId() {
		return productId;
	}
	public void setProductId(Long productId) {
		this.productId = productId;
	}
	public String getComment() {
		return comment;
	}
	public void setComment(String comment) {
		this.comment = comment;
	}
	public Double getRating() {
		return rating;
	}
	public void setRating(Double rating) {
		this.rating = rating;
	}
	@NotBlank(message = "Comment cannot be blank.")
	private String comment;
	@NotNull(message = "Rating is required")
	
	private Double rating;
}
