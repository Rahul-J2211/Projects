package com.jvlcode.jvlcart.dto;

public class OrderCreated {
	private String referenceId;

	public String getReferenceId() {
		return referenceId;
	}

	public void setReferenceId(String referenceId) {
		this.referenceId = referenceId;
	}

	public OrderCreated(String referenceId) {
		super();
		this.referenceId = referenceId;
	}
	
}
