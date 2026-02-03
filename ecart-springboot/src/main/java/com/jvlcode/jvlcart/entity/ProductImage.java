package com.jvlcode.jvlcart.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;

@Entity
public class ProductImage {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Long Id;
	
	private String publicId;
	private String url;
	public Long getId() {
		return Id;
	}
	public ProductImage() {
		super();
		// TODO Auto-generated constructor stub
	}
	public void setId(Long id) {
		Id = id;
	}
	public String getPublicId() {
		return publicId;
	}
	public void setPublicId(String publicId) {
		this.publicId = publicId;
	}
	public String getUrl() {
		return url;
	}
	public void setUrl(String url) {
		this.url = url;
	}
	public ProductImage(Long id, String publicId, String url) {
		super();
		Id = id;
		this.publicId = publicId;
		this.url = url;
	}
	@ManyToOne()
	@JoinColumn(name = "product_id")
	private Product product;
	
	public ProductImage(String url, Product product) {
		// TODO Auto-generated constructor stub
		this.url = "/uploads"+url;
		this.publicId = url;
		this.product = product;
	}
	
	
}
